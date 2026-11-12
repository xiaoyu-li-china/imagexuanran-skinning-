/**
 * animation.ts —— 动画片段加载、关键帧采样与最小动画状态机
 *
 * 支持两种关键帧：
 *   TRS 帧：{ time, translation:[x,y,z], rotation:[x,y,z,w], scale:[x,y,z] }
 *   矩阵帧：{ time, matrix:[16 个列主序数] }（采样时先分解为 TRS 再插值）
 *
 * 状态机在 idle / walk（及任意已加载片段）之间切换，切换时以秒级时间驱动
 * 权重做 Crossfade，切换瞬间新片段权重为 0，因此不会产生姿态跳变。
 */
import {
  Mat4,
  Quat,
  Vec3,
  mat4FromTRS,
  mat4Decompose,
  quatSlerp,
  quatNormalize,
  vec3Lerp,
  smoothstep,
  clamp,
} from './mat4.js';
import { Skeleton, SkeletonParseError } from './skeleton.js';

export interface TrsKeyframe {
  time: number;
  translation?: number[];
  rotation?: number[]; // [x,y,z,w]
  scale?: number[];
}

export interface MatrixKeyframe {
  time: number;
  matrix: number[];
}

export type Keyframe = TrsKeyframe | MatrixKeyframe;

export interface TrackData {
  /** 骨骼名或骨骼下标 */
  bone: string | number;
  keyframes: Keyframe[];
}

export interface ClipData {
  name: string;
  duration: number;
  tracks: TrackData[];
}

/** 采样后的局部 TRS 姿态 */
export interface TrsPose {
  translation: Vec3;
  rotation: Quat;
  scale: Vec3;
}

interface ResolvedTrack {
  boneIndex: number;
  times: number[];
  frames: TrsKeyframe[]; // 矩阵帧在加载时已统一分解为 TRS
}

export class AnimationClipError extends Error {
  constructor(message: string) {
    super(`[动画片段错误] ${message}`);
    this.name = 'AnimationClipError';
  }
}

export class AnimationClip {
  readonly name: string;
  readonly duration: number;
  private readonly tracks: ResolvedTrack[];

  private constructor(name: string, duration: number, tracks: ResolvedTrack[]) {
    this.name = name;
    this.duration = duration;
    this.tracks = tracks;
  }

  static parse(data: ClipData, skeleton: Skeleton): AnimationClip {
    if (data === null || typeof data !== 'object') {
      throw new AnimationClipError('片段数据必须是对象');
    }
    if (typeof data.name !== 'string' || data.name.trim() === '') {
      throw new AnimationClipError('片段 name 必须是非空字符串');
    }
    if (typeof data.duration !== 'number' || !Number.isFinite(data.duration) || data.duration <= 0) {
      throw new AnimationClipError(`片段 "${data.name}" 的 duration 必须是正数，实际为 ${String(data.duration)}`);
    }
    if (!Array.isArray(data.tracks)) {
      throw new AnimationClipError(`片段 "${data.name}" 必须包含 tracks 数组`);
    }

    const usedBones = new Set<number>();
    const tracks: ResolvedTrack[] = data.tracks.map((track, ti) => {
      if (track === null || typeof track !== 'object' || !Array.isArray(track.keyframes)) {
        throw new AnimationClipError(`片段 "${data.name}" 第 ${ti} 条轨道结构非法`);
      }
      const boneIndex =
        typeof track.bone === 'number'
          ? track.bone
          : skeleton.getBoneIndex(track.bone); // 骨骼不存在时抛 SkeletonParseError
      if (boneIndex < 0 || boneIndex >= skeleton.boneCount) {
        throw new AnimationClipError(
          `片段 "${data.name}" 轨道 ${ti} 的骨骼下标 ${boneIndex} 越界（共 ${skeleton.boneCount} 根骨骼）`,
        );
      }
      if (usedBones.has(boneIndex)) {
        throw new AnimationClipError(
          `片段 "${data.name}" 中骨骼 "${skeleton.nodes[boneIndex].name}" 存在多条轨道，每根骨骼每片段至多一条`,
        );
      }
      usedBones.add(boneIndex);

      if (track.keyframes.length === 0) {
        throw new AnimationClipError(`片段 "${data.name}" 轨道 ${ti} 的关键帧为空`);
      }

      const times: number[] = [];
      const frames: TrsKeyframe[] = [];
      let prevTime = -Infinity;
      track.keyframes.forEach((kf, ki) => {
        if (
          kf === null ||
          typeof kf !== 'object' ||
          typeof kf.time !== 'number' ||
          !Number.isFinite(kf.time) ||
          kf.time < 0
        ) {
          throw new AnimationClipError(
            `片段 "${data.name}" 轨道 ${ti} 第 ${ki} 帧的 time 必须是 >= 0 的有限数`,
          );
        }
        if (kf.time <= prevTime) {
          throw new AnimationClipError(
            `片段 "${data.name}" 轨道 ${ti} 关键帧时间必须严格递增，第 ${ki} 帧 time=${kf.time}`,
          );
        }
        prevTime = kf.time;
        times.push(kf.time);

        if ('matrix' in kf) {
          const mk = kf as MatrixKeyframe;
          if (!Array.isArray(mk.matrix) || mk.matrix.length !== 16) {
            throw new AnimationClipError(`片段 "${data.name}" 轨道 ${ti} 第 ${ki} 帧 matrix 必须是长度 16 的数组`);
          }
          const decomposed = mat4Decompose(new Float32Array(mk.matrix) as Mat4);
          frames.push({
            time: mk.time,
            translation: [...decomposed.translation],
            rotation: [...decomposed.rotation],
            scale: [...decomposed.scale],
          });
        } else {
          const tk = kf as TrsKeyframe;
          const checkVec = (v: number[] | undefined, len: number, field: string): number[] => {
            if (v === undefined) return v as unknown as number[];
            if (!Array.isArray(v) || v.length !== len || v.some((x) => typeof x !== 'number' || !Number.isFinite(x))) {
              throw new AnimationClipError(
                `片段 "${data.name}" 轨道 ${ti} 第 ${ki} 帧的 ${field} 必须是 ${len} 个有限数`,
              );
            }
            return v;
          };
          checkVec(tk.translation, 3, 'translation');
          checkVec(tk.scale, 3, 'scale');
          if (tk.rotation !== undefined) {
            checkVec(tk.rotation, 4, 'rotation');
            const n = Math.hypot(...tk.rotation);
            if (n < 1e-8) {
              throw new AnimationClipError(
                `片段 "${data.name}" 轨道 ${ti} 第 ${ki} 帧的旋转四元数为零向量`,
              );
            }
          }
          frames.push({ time: tk.time, translation: tk.translation, rotation: tk.rotation, scale: tk.scale });
        }
      });

      return { boneIndex, times, frames };
    });

    return new AnimationClip(data.name, data.duration, tracks);
  }

  /**
   * 在 time（秒，自动循环到 [0, duration)）处采样。
   * 没有轨道的骨骼返回 undefined，由上层用绑定姿态补齐。
   */
  sample(time: number, fallback: (boneIndex: number) => TrsPose): Map<number, TrsPose> {
    const wrapped = ((time % this.duration) + this.duration) % this.duration;
    const result = new Map<number, TrsPose>();

    for (const track of this.tracks) {
      const { times, frames } = track;
      let pose: TrsPose;

      if (wrapped <= times[0]) {
        pose = this.evalFrame(frames[0], frames[0], 0, fallback(track.boneIndex));
      } else if (wrapped >= times[times.length - 1]) {
        const last = frames[frames.length - 1];
        pose = this.evalFrame(last, last, 0, fallback(track.boneIndex));
      } else {
        // 二分查找相邻关键帧
        let lo = 0;
        let hi = times.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (times[mid] <= wrapped) lo = mid;
          else hi = mid;
        }
        const span = times[hi] - times[lo];
        const alpha = span > 0 ? (wrapped - times[lo]) / span : 0;
        pose = this.evalFrame(frames[lo], frames[hi], alpha, fallback(track.boneIndex));
      }
      result.set(track.boneIndex, pose);
    }
    return result;
  }

  /** 两帧之间做 TRS 插值；缺省通道回退到绑定姿态 */
  private evalFrame(a: TrsKeyframe, b: TrsKeyframe, alpha: number, bind: TrsPose): TrsPose {
    const ta = a.translation ?? bind.translation;
    const tb = b.translation ?? bind.translation;
    const sa = a.scale ?? bind.scale;
    const sb = b.scale ?? bind.scale;
    const ra = a.rotation ? quatNormalize(new Float32Array(a.rotation) as Quat) : bind.rotation;
    const rb = b.rotation ? quatNormalize(new Float32Array(b.rotation) as Quat) : bind.rotation;

    return {
      translation: vec3Lerp(ta, tb, alpha),
      rotation: quatSlerp(ra, rb, alpha),
      scale: vec3Lerp(sa, sb, alpha),
    };
  }
}

interface ActiveSource {
  clip: AnimationClip;
  time: number;
  weight: number;
  /** 淡入：权重从 0 平滑到 1 */
  fadeIn?: { elapsed: number; duration: number };
  /** 淡出：权重从 fadeStart 平滑到 0 */
  fadeOut?: { elapsed: number; duration: number; fadeStart: number };
}

export interface StateMachineSnapshot {
  primary: string | null;
  sources: { name: string; weight: number; time: number }[];
}

/**
 * 最小动画状态机。
 * 权重模型：新片段以权重 0 淡入（保证切换瞬间姿态不变，无跳变），
 * 旧片段同时淡出；每帧对所有权重做归一化后混合 TRS，再重组局部矩阵。
 */
export class AnimationStateMachine {
  private readonly skeleton: Skeleton;
  private readonly clips = new Map<string, AnimationClip>();
  private readonly bindLocalTrs: TrsPose[];
  private sources: ActiveSource[] = [];
  private defaultFadeSeconds = 0.6;

  constructor(skeleton: Skeleton) {
    this.skeleton = skeleton;
    // 缓存绑定姿态局部矩阵的 TRS 分解，作为缺失轨道的回退
    this.bindLocalTrs = skeleton.nodes.map((n) => mat4Decompose(n.local));
  }

  loadClip(clip: AnimationClip): void {
    if (this.clips.has(clip.name)) {
      throw new AnimationClipError(`片段 "${clip.name}" 已加载，名称不得重复`);
    }
    this.clips.set(clip.name, clip);
  }

  hasClip(name: string): boolean {
    return this.clips.has(name);
  }

  setDefaultFade(seconds: number): void {
    if (!(seconds > 0) || !Number.isFinite(seconds)) {
      throw new Error('Crossfade 时长必须是正数（秒）');
    }
    this.defaultFadeSeconds = seconds;
  }

  /**
   * 切换到目标片段。
   * @param name 片段名
   * @param fadeSeconds Crossfade 时长（秒），缺省用默认值
   */
  play(name: string, fadeSeconds?: number): void {
    const clip = this.clips.get(name);
    if (!clip) {
      throw new AnimationClipError(`无法切换：片段 "${name}" 尚未加载`);
    }
    const fade = fadeSeconds ?? this.defaultFadeSeconds;
    if (!(fade > 0) || !Number.isFinite(fade)) {
      throw new Error('Crossfade 时长必须是正数（秒）');
    }

    // 已经是唯一主片段：不重复触发
    if (this.sources.length === 1 && this.sources[0].clip.name === name && !this.sources[0].fadeOut) {
      return;
    }
    // 切换过程中再次请求同一片段：若该片段正在淡入，保持现状即可
    if (this.sources.some((s) => s.clip.name === name && s.fadeIn && !s.fadeOut)) {
      return;
    }

    // 现有片段全部转入淡出（记录当前权重作为淡出起点）
    for (const s of this.sources) {
      if (!s.fadeOut) {
        s.fadeOut = { elapsed: 0, duration: fade, fadeStart: s.weight };
      }
      s.fadeIn = undefined;
    }
    this.sources.push({ clip, time: 0, weight: 0, fadeIn: { elapsed: 0, duration: fade } });
  }

  /**
   * 推进状态机。
   * @param dtSeconds 距上一帧的秒数
   * @returns 当前帧每个节点的局部矩阵（可直接喂给 Skeleton.computeFK）
   */
  update(dtSeconds: number): Mat4[] {
    if (!Number.isFinite(dtSeconds) || dtSeconds < 0) {
      throw new Error(`update 的 dt 必须是非负有限数，实际为 ${String(dtSeconds)}`);
    }

    // 1. 推进时间与淡入淡出
    for (const s of this.sources) {
      s.time = (s.time + dtSeconds) % s.clip.duration;
      if (s.fadeIn) {
        s.fadeIn.elapsed += dtSeconds;
        const p = clamp(s.fadeIn.elapsed / s.fadeIn.duration, 0, 1);
        s.weight = smoothstep(0, 1, p);
        if (p >= 1) s.fadeIn = undefined;
      }
      if (s.fadeOut) {
        s.fadeOut.elapsed += dtSeconds;
        const p = clamp(s.fadeOut.elapsed / s.fadeOut.duration, 0, 1);
        s.weight = s.fadeOut.fadeStart * (1 - smoothstep(0, 1, p));
      }
    }
    // 清理淡出完成的源
    this.sources = this.sources.filter((s) => !(s.fadeOut && s.fadeOut.elapsed >= s.fadeOut.duration));

    // 2. 权重归一化（所有源同时从 0 起步的极端情况下退化为绑定姿态）
    const totalWeight = this.sources.reduce((sum, s) => sum + s.weight, 0);
    const normalized = this.sources.map((s) => (totalWeight > 1e-8 ? s.weight / totalWeight : 0));

    // 3. 采样每个源并按归一化权重混合 TRS
    const boneCount = this.skeleton.boneCount;
    const blended: TrsPose[] = [];
    for (let bone = 0; bone < boneCount; bone++) {
      blended.push(this.blendBone(bone, normalized));
    }

    // 4. TRS 重组为局部矩阵
    return blended.map((p) => mat4FromTRS(p.translation, p.rotation, p.scale));
  }

  /** 对单根骨骼混合所有活动片段；无有效源时使用绑定姿态 */
  private blendBone(bone: number, normalized: number[]): TrsPose {
    const bind = this.bindLocalTrs[bone];
    let acc: TrsPose | null = null;
    let cumWeight = 0;

    for (let i = 0; i < this.sources.length; i++) {
      const w = normalized[i];
      if (w <= 1e-8) continue;
      const pose =
        this.sources[i].clip.sample(this.sources[i].time, (b) => this.bindLocalTrs[b]).get(bone) ?? bind;

      if (acc === null) {
        acc = {
          translation: new Float32Array(pose.translation),
          rotation: new Float32Array(pose.rotation),
          scale: new Float32Array(pose.scale),
        };
      } else {
        // 累积 slerp：在已累积姿态与当前姿态之间按相对权重插值
        const localT = w / (cumWeight + w);
        acc.translation = vec3Lerp(acc.translation, pose.translation, localT);
        acc.scale = vec3Lerp(acc.scale, pose.scale, localT);
        acc.rotation = quatSlerp(acc.rotation, pose.rotation, localT);
      }
      cumWeight += w;
    }

    if (acc === null) {
      return {
        translation: new Float32Array(bind.translation),
        rotation: new Float32Array(bind.rotation),
        scale: new Float32Array(bind.scale),
      };
    }
    return acc;
  }

  snapshot(): StateMachineSnapshot {
    let primary: string | null = null;
    let best = -1;
    for (const s of this.sources) {
      if (s.weight > best) {
        best = s.weight;
        primary = s.clip.name;
      }
    }
    return {
      primary,
      sources: this.sources.map((s) => ({ name: s.clip.name, weight: s.weight, time: s.time })),
    };
  }

  get activeSourceCount(): number {
    return this.sources.length;
  }
}

export { SkeletonParseError };
