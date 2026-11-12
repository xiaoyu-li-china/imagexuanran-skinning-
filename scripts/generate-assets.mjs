/**
 * generate-assets.mjs —— 程序化生成骨骼与动画 JSON 资产（纯 Node 运行，无依赖）
 * 输出：
 *   src/assets/skeleton.json   5 节骨骼链 root -> bone1..bone4，每节局部平移 (0,1,0)
 *   src/assets/animations.json idle / walk 两个片段，关键帧同时包含旋转(四元数)与缩放
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = join(here, '..', 'src', 'assets');
mkdirSync(outDir, { recursive: true });

/** 列主序平移矩阵 */
function translationMat(x, y, z) {
  const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
  return m;
}

/** 绕 Z 轴旋转 angle 弧度的四元数 [x,y,z,w] */
function quatZ(angle) {
  return [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];
}

// ---------- 骨骼层级 ----------
const boneNames = ['root', 'bone1', 'bone2', 'bone3', 'bone4'];
const skeleton = {
  formatVersion: 1,
  nodes: boneNames.map((name, i) => ({
    name,
    parent: i - 1,
    localMatrix: translationMat(0, i === 0 ? 0 : 1, 0),
  })),
};
writeFileSync(join(outDir, 'skeleton.json'), JSON.stringify(skeleton, null, 2));

// ---------- 动画片段 ----------
// 关键帧必须覆盖正弦波的极值点（过零点上的关键帧无法表达动作）
const DURATION = 2.0;
const TIMES = [0, 0.5, 1.0, 1.5, 2.0];

/**
 * 生成一条轨道
 * @param {number} boneIndex 骨骼下标
 * @param {(t:number)=>{angle:number,sx:number,sy:number,sz:number,tx:number}} pose
 */
function track(boneIndex, pose) {
  return {
    bone: boneIndex,
    keyframes: TIMES.map((t) => {
      const p = pose(t);
      return {
        time: t,
        translation: [p.tx, 0, 0],
        rotation: quatZ(p.angle),
        scale: [p.sx, p.sy, p.sz],
      };
    }),
  };
}

// idle：小幅左右摆动 + 轻微呼吸缩放（幅度随骨骼层级递增）
const idleTracks = boneNames.map((_, i) => {
  const amp = 0.06 * (i + 1);
  return track(i, (t) => {
    const wave = Math.sin((2 * Math.PI * t) / DURATION);
    return {
      angle: amp * wave,
      sx: 1 + 0.04 * wave,
      sy: 1 - 0.03 * wave,
      sz: 1 + 0.04 * wave,
      tx: 0.02 * wave,
    };
  });
});

// walk：大幅度交替摆动 + 明显缩放脉冲，相邻骨骼相位相反
const walkTracks = boneNames.map((_, i) => {
  const amp = 0.22 * (i + 1);
  const phase = i % 2 === 0 ? 0 : Math.PI;
  return track(i, (t) => {
    const wave = Math.sin((2 * Math.PI * t) / DURATION + phase);
    return {
      angle: amp * wave,
      sx: 1 + 0.12 * wave,
      sy: 1 - 0.1 * wave,
      sz: 1 + 0.12 * wave,
      tx: 0.1 * wave,
    };
  });
});

const animations = {
  formatVersion: 1,
  clips: [
    { name: 'idle', duration: DURATION, tracks: idleTracks },
    { name: 'walk', duration: DURATION, tracks: walkTracks },
  ],
};
writeFileSync(join(outDir, 'animations.json'), JSON.stringify(animations, null, 2));

console.log(`已生成 ${boneNames.length} 根骨骼与 2 个动画片段到 ${outDir}`);
