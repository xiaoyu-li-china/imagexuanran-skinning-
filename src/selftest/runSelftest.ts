/**
 * runSelftest.ts —— 骨骼动画管线最小自测
 *
 * 覆盖：
 *  A. 骨骼 JSON 解析与 FK 全局矩阵（含绑定姿态坐标解析解）
 *  B. 非法数据：重名 / 循环引用 / 自引用 / 父节点越界 / 多根 / 坏矩阵 / 坏 JSON
 *  C. idle/walk 片段加载与 Crossfade：切换瞬间无跳变、中点权重 0.5/0.5、
 *     中间插值帧的特定骨骼全局坐标输出，且与独立参考解析解对比
 *  D. 全流程所有局部/全局/offset 矩阵非奇异（行列式检查）
 *  E. uniform 预算与分批规划：强制低预算触发多批次，校验覆盖完整性
 *  F. 矩阵分解/重组往返一致性
 *
 * Node 下：npx tsc -p tsconfig.test.json && node dist-test/selftest/runSelftest.js
 * 浏览器下：main.ts 调用 runSelftest()，结果同步打印到控制台。
 */
import {
  mat4Multiply,
  mat4Determinant,
  mat4TransformPoint,
  mat4FromTRS,
  mat4Decompose,
  quatSlerp,
  quatFromAxisAngle,
  quatNormalize,
  vec3Lerp,
  mat4ToString,
} from '../core/mat4.js';
import { Skeleton, SkeletonData } from '../core/skeleton.js';
import {
  AnimationClip,
  AnimationStateMachine,
  ClipData,
  TrackData,
  TrsKeyframe,
  TrsPose,
} from '../core/animation.js';
import {
  BASE_VERTEX_UNIFORM_VECTORS,
  BONE_LIMIT_SAFETY_FACTOR,
  computeUniformBudget,
  planBatches,
} from '../core/batching.js';

export interface SelftestResult {
  passed: number;
  failed: number;
  lines: string[];
}

class Reporter {
  passed = 0;
  failed = 0;
  lines: string[] = [];

  log(msg: string): void {
    this.lines.push(msg);
    console.log(msg);
  }

  check(name: string, condition: boolean, detail = ''): void {
    if (condition) {
      this.passed++;
      this.log(`  ✅ ${name}${detail ? ` —— ${detail}` : ''}`);
    } else {
      this.failed++;
      this.log(`  ❌ ${name}${detail ? ` —— ${detail}` : ''}`);
    }
  }

  section(title: string): void {
    this.log('');
    this.log(`=== ${title} ===`);
  }
}

// ---------------- 独立参考实现（仅直接读 JSON + 调用 mat4 数学库，不碰生产采样/状态机） ----------------

function referenceFramePose(kf: TrsKeyframe, bind: TrsPose): TrsPose {
  return {
    translation: new Float32Array(kf.translation ?? bind.translation),
    rotation: kf.rotation ? quatNormalize(new Float32Array(kf.rotation)) : new Float32Array(bind.rotation),
    scale: new Float32Array(kf.scale ?? bind.scale),
  };
}

/** 在两个关键帧之间做线性 / slerp 插值（与生产代码独立编写的参考版本） */
function referenceSampleTrack(track: TrackData, time: number, bind: TrsPose): TrsPose {
  const frames = track.keyframes as TrsKeyframe[];
  const times = frames.map((f) => f.time);
  if (time <= times[0]) return referenceFramePose(frames[0], bind);
  if (time >= times[times.length - 1]) return referenceFramePose(frames[times.length - 1], bind);
  let hi = 1;
  while (times[hi] <= time) hi++;
  const lo = hi - 1;
  const alpha = (time - times[lo]) / (times[hi] - times[lo]);
  const a = referenceFramePose(frames[lo], bind);
  const b = referenceFramePose(frames[hi], bind);
  return {
    translation: vec3Lerp(a.translation, b.translation, alpha),
    rotation: quatSlerp(a.rotation, b.rotation, alpha),
    scale: vec3Lerp(a.scale, b.scale, alpha),
  };
}

/** 权重归一化后混合多个 TRS 姿态（与状态机算法等价的独立实现） */
function referenceBlend(entries: { pose: TrsPose; weight: number }[]): TrsPose {
  const total = entries.reduce((s, e) => s + e.weight, 0);
  let acc: TrsPose | null = null;
  let cum = 0;
  for (const e of entries) {
    const w = total > 0 ? e.weight / total : 0;
    if (w <= 1e-9) continue;
    if (acc === null) {
      acc = {
        translation: new Float32Array(e.pose.translation),
        rotation: new Float32Array(e.pose.rotation),
        scale: new Float32Array(e.pose.scale),
      };
    } else {
      const t = w / (cum + w);
      acc.translation = vec3Lerp(acc.translation, e.pose.translation, t);
      acc.scale = vec3Lerp(acc.scale, e.pose.scale, t);
      acc.rotation = quatSlerp(acc.rotation, e.pose.rotation, t);
    }
    cum += w;
  }
  if (acc === null) throw new Error('参考混合失败：无有效权重');
  return acc;
}

/** 读取 JSON 资产：Node 用 fs，浏览器由调用方注入 loadJson */
export interface SelftestAssets {
  skeletonData: SkeletonData;
  animationsData: { clips: ClipData[] };
}

export async function loadAssetsForSelftest(): Promise<SelftestAssets> {
  const skeletonData = (await fetchJson('/src/assets/skeleton.json')) as SkeletonData;
  const animationsData = (await fetchJson('/src/assets/animations.json')) as { clips: ClipData[] };
  return { skeletonData, animationsData };
}

async function fetchJson(url: string): Promise<unknown> {
  // 浏览器环境才有 window；Node 下走 fs（Node 18+ 虽自带全局 fetch，但无法解析相对 URL）
  const isBrowser = typeof (globalThis as { window?: unknown }).window !== 'undefined';
  if (isBrowser && typeof fetch === 'function') {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} 失败：HTTP ${res.status}`);
    return res.json();
  }
  // Node（编译后位于 dist-test/selftest/，资产在工程根 src/assets/）
  const { readFileSync } = await import(/* @vite-ignore */ hidden('node:fs'));
  const { fileURLToPath } = await import(/* @vite-ignore */ hidden('node:url'));
  const { dirname, resolve } = await import(/* @vite-ignore */ hidden('node:path'));
  const here = dirname(fileURLToPath(import.meta.url));
  const rel = url.replace(/^\//, '');
  const candidate = resolve(here, '..', '..', rel);
  return JSON.parse(readFileSync(candidate, 'utf-8'));
}

/** 阻止打包器静态分析出 Node 内置模块说明符（该分支在浏览器中永不执行） */
function hidden(spec: string): string {
  return spec;
}

export async function runSelftest(assets?: SelftestAssets): Promise<SelftestResult> {
  const r = new Reporter();
  r.log('################ 骨骼动画管线自测开始 ################');

  const { skeletonData, animationsData } = assets ?? (await loadAssetsForSelftest());

  // ---------- A. 解析与 FK ----------
  r.section('A. 骨骼 JSON 解析与正向运动学（FK）');
  const skeleton = Skeleton.parse(skeletonData);
  r.check('解析 5 根骨骼', skeleton.boneCount === 5, `实际 ${skeleton.boneCount} 根`);
  r.check('根节点为 root 且下标 0', skeleton.rootIndex === 0 && skeleton.nodes[0].name === 'root');
  r.check('遍历顺序满足父先子后', skeleton.traverseOrder.join(',') === '0,1,2,3,4');

  const bindGlobals = skeleton.computeFK();
  const bindOk = [0, 1, 2, 3, 4].every((i) => {
    const p = mat4TransformPoint(bindGlobals[i], [0, 0, 0]);
    return Math.abs(p[0]) < 1e-6 && Math.abs(p[1] - i) < 1e-6 && Math.abs(p[2]) < 1e-6;
  });
  r.check('FK 绑定姿态全局坐标：bone i 位于 (0, i, 0)', bindOk);
  r.log('  bone4 绑定姿态全局矩阵：');
  r.log(mat4ToString(bindGlobals[4]));

  // ---------- B. 非法数据 ----------
  r.section('B. 非法骨骼数据校验（均须抛出可读错误）');
  const base: SkeletonData = JSON.parse(JSON.stringify(skeletonData));
  const expectParseError = (name: string, mutate: (d: SkeletonData) => void): void => {
    const d = JSON.parse(JSON.stringify(base)) as SkeletonData;
    mutate(d);
    let caught = '';
    try {
      Skeleton.parse(d);
    } catch (e) {
      caught = (e as Error).message;
    }
    r.check(name, caught.includes('[骨骼数据错误]'), caught || '未抛出错误');
  };

  expectParseError('重名节点被拒绝', (d) => {
    d.nodes[2].name = d.nodes[1].name;
  });
  expectParseError('循环引用被拒绝（成环节点与根断开，报不可达）', (d) => {
    d.nodes[2].parent = 3;
  });
  expectParseError('根自身成环被拒绝（无 parent=-1 节点）', (d) => {
    d.nodes[0].parent = 1;
    d.nodes[1].parent = 0;
  });
  expectParseError('自引用被拒绝', (d) => {
    d.nodes[1].parent = 1;
  });
  expectParseError('父节点下标越界被拒绝', (d) => {
    d.nodes[1].parent = 99;
  });
  expectParseError('多个根节点被拒绝', (d) => {
    d.nodes[2].parent = -1;
  });
  expectParseError('非仿射矩阵被拒绝', (d) => {
    d.nodes[0].localMatrix[15] = 2;
  });
  expectParseError('NaN 矩阵被拒绝', (d) => {
    d.nodes[0].localMatrix[0] = Number.NaN;
  });
  expectParseError('矩阵长度错误被拒绝', (d) => {
    d.nodes[0].localMatrix.pop();
  });

  let badJson = '';
  try {
    Skeleton.parse('{ not valid json');
  } catch (e) {
    badJson = (e as Error).message;
  }
  r.check('坏 JSON 字符串被拒绝', badJson.includes('JSON 解析失败'), badJson);

  // ---------- C. 动画与 Crossfade ----------
  r.section('C. 动画片段加载、状态机与 Crossfade');
  const clips = animationsData.clips.map((c) => AnimationClip.parse(c, skeleton));
  r.check('加载 idle / walk 两个片段', clips.length === 2 && clips[0].name === 'idle' && clips[1].name === 'walk');

  let unknownBone = '';
  try {
    AnimationClip.parse(
      { name: 'bad', duration: 1, tracks: [{ bone: 'ghost', keyframes: [{ time: 0, scale: [1, 1, 1] }] }] },
      skeleton,
    );
  } catch (e) {
    unknownBone = (e as Error).message;
  }
  r.check('引用不存在骨骼的片段被拒绝', unknownBone.includes('ghost'), unknownBone);

  let badTimes = '';
  try {
    AnimationClip.parse(
      {
        name: 'bad2',
        duration: 1,
        tracks: [{ bone: 0, keyframes: [{ time: 0.5, scale: [1, 1, 1] }, { time: 0.2, scale: [1, 1, 1] }] }],
      },
      skeleton,
    );
  } catch (e) {
    badTimes = (e as Error).message;
  }
  r.check('关键帧时间非递增被拒绝', badTimes.includes('严格递增'), badTimes);

  const fsm = new AnimationStateMachine(skeleton);
  clips.forEach((c) => fsm.loadClip(c));
  fsm.setDefaultFade(0.6);
  fsm.play('idle', 0.6);

  // 先让 idle 完成 0.6s 淡入（权重到 1），再测试切换
  for (let k = 0; k < 60; k++) fsm.update(0.01);

  // C1. 切换瞬间无跳变：记录切换前姿态，切换后以极小 dt 推进
  const idlePoseBefore = fsm.update(0);
  fsm.play('walk', 0.6);
  const poseAtSwitch = fsm.update(1e-9);
  let jump = 0;
  for (let i = 0; i < idlePoseBefore.length; i++) {
    for (let j = 0; j < 16; j++) jump = Math.max(jump, Math.abs(idlePoseBefore[i][j] - poseAtSwitch[i][j]));
  }
  r.check('切换瞬间姿态无跳变（最大矩阵元素差 < 1e-6）', jump < 1e-6, `max diff=${jump.toExponential(2)}`);

  // C2. 再推进 0.3s 到达 fade 中点：idle 淡出到 0.5、walk 淡入到 0.5
  for (let k = 0; k < 29; k++) fsm.update(0.01);
  const midLocals = fsm.update(0.01 - 1e-9);
  const snap = fsm.snapshot();
  const wIdle = snap.sources.find((s) => s.name === 'idle')?.weight ?? -1;
  const wWalk = snap.sources.find((s) => s.name === 'walk')?.weight ?? -1;
  r.check(
    'Crossfade 中点权重 idle≈walk≈0.5',
    Math.abs(wIdle - 0.5) < 1e-3 && Math.abs(wWalk - 0.5) < 1e-3,
    `idle=${wIdle.toFixed(4)} walk=${wWalk.toFixed(4)}`,
  );

  // C3. 独立参考解：不经过 AnimationClip/状态机，直接从 JSON 关键帧插值并混合
  const bindTrs = skeleton.nodes.map((n) => mat4Decompose(n.local));
  const idleSrc = snap.sources.find((s) => s.name === 'idle')!;
  const walkSrc = snap.sources.find((s) => s.name === 'walk')!;
  const idleTrack0 = animationsData.clips[0].tracks.find((t) => t.bone === 0)!;
  const walkTrack0 = animationsData.clips[1].tracks.find((t) => t.bone === 0)!;
  const refIdle = referenceSampleTrack(idleTrack0, idleSrc.time, bindTrs[0]);
  const refWalk = referenceSampleTrack(walkTrack0, walkSrc.time, bindTrs[0]);
  const refRoot = referenceBlend([
    { pose: refIdle, weight: idleSrc.weight },
    { pose: refWalk, weight: walkSrc.weight },
  ]);

  const midRootDecomp = mat4Decompose(midLocals[0]);
  const angleErr = Math.min(
    Math.hypot(...midRootDecomp.rotation.map((v, k) => v - refRoot.rotation[k])),
    Math.hypot(...midRootDecomp.rotation.map((v, k) => v + refRoot.rotation[k])), // 四元数双覆盖
  );
  const scaleErr = Math.hypot(
    midRootDecomp.scale[0] - refRoot.scale[0],
    midRootDecomp.scale[1] - refRoot.scale[1],
    midRootDecomp.scale[2] - refRoot.scale[2],
  );
  const translErr = Math.hypot(
    midRootDecomp.translation[0] - refRoot.translation[0],
    midRootDecomp.translation[1] - refRoot.translation[1],
    midRootDecomp.translation[2] - refRoot.translation[2],
  );
  r.check(
    '中间帧 root 旋转与独立参考解一致（四元数误差 < 1e-4）',
    angleErr < 1e-4,
    `err=${angleErr.toExponential(2)}；idle 本地 t=${idleSrc.time.toFixed(3)}s，walk 本地 t=${walkSrc.time.toFixed(3)}s`,
  );
  r.check('中间帧 root 缩放与独立参考解一致', scaleErr < 1e-4, `err=${scaleErr.toExponential(2)}`);
  r.check('中间帧 root 平移与独立参考解一致', translErr < 1e-4, `err=${translErr.toExponential(2)}`);
  r.log(
    `  参考信息：关键帧时间点 [${(idleTrack0.keyframes as TrsKeyframe[]).map((f) => f.time).join(', ')}]，` +
      `参考缩放 sx=${refRoot.scale[0].toFixed(5)}，实际 sx=${midRootDecomp.scale[0].toFixed(5)}`,
  );

  // C4. 输出中间插值帧特定骨骼（bone2 / bone4）的全局坐标
  const midGlobals = skeleton.computeFK(midLocals);
  const bone2Pos = mat4TransformPoint(midGlobals[2], [0, 0, 0]);
  const bone4Pos = mat4TransformPoint(midGlobals[4], [0, 0, 0]);
  r.log('  —— Crossfade 中间帧（t≈0.3s，权重 0.5/0.5）骨骼全局坐标 ——');
  r.log(`  bone2 全局位置 = (${Array.from(bone2Pos, (v) => v.toFixed(4)).join(', ')})`);
  r.log(`  bone4 全局位置 = (${Array.from(bone4Pos, (v) => v.toFixed(4)).join(', ')})`);
  r.check('中间帧 bone4 偏离绑定位置（动画确实生效）', Math.hypot(bone4Pos[0], bone4Pos[1] - 4) > 1e-3);
  r.log('  bone2 中间帧全局矩阵：');
  r.log(mat4ToString(midGlobals[2]));

  // ---------- D. 非奇异检查 ----------
  r.section('D. 插值矩阵非奇异校验（行列式）');
  let singularCount = 0;
  let minAbsDet = Infinity;
  const fsm2 = new AnimationStateMachine(skeleton);
  clips.forEach((c) => fsm2.loadClip(c));
  fsm2.play('idle', 0.6);
  fsm2.play('walk', 0.6);
  for (let frame = 0; frame < 60; frame++) {
    const locals = fsm2.update(0.01);
    const globals = skeleton.computeFK(locals);
    for (let i = 0; i < skeleton.boneCount; i++) {
      const dLocal = Math.abs(mat4Determinant(locals[i]));
      const dGlobal = Math.abs(mat4Determinant(globals[i]));
      const offset = mat4Multiply(globals[i], skeleton.inverseBind[i]);
      const dOffset = Math.abs(mat4Determinant(offset));
      minAbsDet = Math.min(minAbsDet, dLocal, dGlobal, dOffset);
      if (dLocal < 1e-8 || dGlobal < 1e-8 || dOffset < 1e-8) singularCount++;
    }
  }
  r.check('60 帧 crossfade 全部局部/全局/offset 矩阵非奇异', singularCount === 0, `奇异矩阵 ${singularCount} 个`);
  r.check('最小行列式绝对值处于合理量级', minAbsDet > 1e-6, `min|det|=${minAbsDet.toExponential(2)}`);

  // ---------- E. uniform 预算与分批 ----------
  r.section('E. MAX_VERTEX_UNIFORM_VECTORS 预算与分批降级');
  const budgetNormal = computeUniformBudget(256, 5);
  r.check(
    '常规预算（256 vectors）无需分批',
    !budgetNormal.needsBatching &&
      budgetNormal.maxBonesPerBatch ===
        Math.floor((Math.floor(256 * BONE_LIMIT_SAFETY_FACTOR) - BASE_VERTEX_UNIFORM_VECTORS) / 4),
    `单批上限 ${budgetNormal.maxBonesPerBatch} 根`,
  );

  const budgetLow = computeUniformBudget(32, 5);
  r.check(
    '强制低预算（32 vectors）触发分批',
    budgetLow.needsBatching && budgetLow.maxBonesPerBatch === 3,
    `单批上限 ${budgetLow.maxBonesPerBatch} 根`,
  );

  // 构造 6 顶点、每顶点 2 骨骼的合成数据，单批 2 根 → 必须拆批
  const synthIndices = new Uint16Array([0, 1, 1, 2, 2, 3, 3, 4, 0, 4, 1, 3]);
  const plan = planBatches(synthIndices, 2, 5, 2);
  r.check('分批数 > 1', plan.batchCount > 1, `拆为 ${plan.batchCount} 批`);
  const covered = new Set<number>();
  let localIdsValid = true;
  for (const b of plan.boneBatches) {
    b.forEach((id) => covered.add(id));
    if (b.length > 2) localIdsValid = false;
  }
  for (const vb of plan.vertexBatches) {
    if (vb.localBoneIds.some((id) => id < 0 || id >= 2)) localIdsValid = false;
  }
  r.check('各批骨骼不超容量且局部 id 合法', localIdsValid);
  r.check('分批覆盖全部 5 根骨骼', covered.size === 5, `覆盖 ${covered.size} 根`);
  r.check(
    '所有顶点都被某个批次处理',
    plan.vertexBatches.reduce((n, vb) => n + (vb.end - vb.start), 0) === 6,
  );

  // ---------- F. 分解重组往返 ----------
  r.section('F. TRS 分解 / 重组往返一致性');
  let roundtripErr = 0;
  for (const locals of [midLocals, skeleton.computeFK()]) {
    for (const m of locals) {
      const d = mat4Decompose(m);
      const rebuilt = mat4FromTRS(d.translation, d.rotation, d.scale);
      for (let i = 0; i < 16; i++) roundtripErr = Math.max(roundtripErr, Math.abs(rebuilt[i] - m[i]));
    }
  }
  r.check('分解后重组矩阵最大元素误差 < 1e-5', roundtripErr < 1e-5, `max err=${roundtripErr.toExponential(2)}`);

  // 附带验证 slerp 端点性质
  const q0 = quatNormalize(new Float32Array([0, 0, 0, 1]));
  const q1 = quatNormalize(quatFromAxisAngle([0, 0, 1], Math.PI / 2));
  const qMid = quatSlerp(q0, q1, 0.5);
  const qMidRef = quatNormalize(quatFromAxisAngle([0, 0, 1], Math.PI / 4));
  const slerpErr = Math.hypot(...qMid.map((v, k) => v - qMidRef[k]));
  r.check('slerp(0.5) 等于半角旋转', slerpErr < 1e-6, `err=${slerpErr.toExponential(2)}`);
  const vLerp = vec3Lerp([0, 0, 0], [2, 4, 6], 0.5);
  r.check('vec3Lerp 中点正确', vLerp[0] === 1 && vLerp[1] === 2 && vLerp[2] === 3);

  r.log('');
  r.log(`################ 自测结束：通过 ${r.passed} 项，失败 ${r.failed} 项 ################`);
  return { passed: r.passed, failed: r.failed, lines: r.lines };
}

// Node CLI 入口
const isNode = typeof (globalThis as { process?: unknown }).process !== 'undefined' &&
  typeof (globalThis as { window?: unknown }).window === 'undefined';
if (isNode) {
  runSelftest()
    .then((result) => {
      if (result.failed > 0) process.exitCode = 1;
    })
    .catch((e) => {
      console.error('自测运行异常：', e);
      process.exitCode = 1;
    });
}
