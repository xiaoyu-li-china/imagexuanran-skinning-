/**
 * meshBuilder.ts —— 程序化生成一条沿 Y 轴的分段方柱蒙皮网格
 *
 * 骨骼链 root(0) -> bone1(1) -> ... -> bone4(4)，相邻骨骼间距 1 个单位。
 * 每段柱体 8 个顶点（上下环各 4 个），段交界处的顶点由相邻两根骨骼
 * 各 0.5 权重共同影响，演示真正的多骨骼线性混合。
 */
import { SkinnedMeshData } from '../webgl/renderer.js';

const SEGMENT_COUNT = 4; // 对应 bone0..bone3 与下一根骨骼之间的段
const SEGMENT_HEIGHT = 1;
const HALF = 0.22;

export function buildSkinnedLimb(): SkinnedMeshData {
  const positions: number[] = [];
  const normals: number[] = [];
  const boneIndices: number[] = [];
  const weights: number[] = [];

  // 四个侧面：normal 分别 ±X / ±Z，每面 4 个顶点（上下各 2 个）
  // 绕序按"从外侧看逆时针"定义（默认 GL_CCW 为正面，与法线方向一致）
  const faces = [
    { n: [1, 0, 0], corners: [[1, 1], [1, -1]] },    // +X 面，z 取 +1/-1
    { n: [-1, 0, 0], corners: [[-1, -1], [-1, 1]] },
    { n: [0, 0, 1], corners: [[1, 1], [-1, 1]] },    // +Z 面，x 取 +1/-1
    { n: [0, 0, -1], corners: [[-1, -1], [1, -1]] },
  ];

  for (let seg = 0; seg < SEGMENT_COUNT; seg++) {
    const y0 = seg * SEGMENT_HEIGHT;
    const y1 = (seg + 1) * SEGMENT_HEIGHT;
    for (const face of faces) {
      const [c0, c1] = face.corners;
      // 每个面：下环 2 顶点 + 上环 2 顶点（三角形顺序绕 normal 逆时针）
      const ring = [
        { y: y0, h: c0 },
        { y: y0, h: c1 },
        { y: y1, h: c1 },
        { y: y1, h: c0 },
      ];
      for (const r of ring) {
        const x = face.n[0] !== 0 ? face.n[0] * HALF : r.h[0] * HALF;
        const z = face.n[2] !== 0 ? face.n[2] * HALF : r.h[1] * HALF;
        positions.push(x, r.y, z);
        normals.push(face.n[0], face.n[1], face.n[2]);

        // 关节处权重混合：段起点归 seg，段终点归 seg+1
        const isTop = r.y === y1;
        const a = isTop ? seg + 1 : seg;
        const b = isTop ? seg : seg + 1;
        boneIndices.push(a, b);
        weights.push(0.5, 0.5);
      }
    }
  }

  const indices: number[] = [];
  for (let seg = 0; seg < SEGMENT_COUNT; seg++) {
    for (let f = 0; f < faces.length; f++) {
      const base = seg * faces.length * 4 + f * 4;
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    boneIndices: new Uint16Array(boneIndices),
    weights: new Float32Array(weights),
    bonesPerVertex: 2,
    indices: new Uint16Array(indices),
    indexCount: indices.length,
  };
}
