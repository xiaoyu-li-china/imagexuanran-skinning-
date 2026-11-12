/**
 * batching.ts —— GPU uniform 预算计算与蒙皮分批规划（纯逻辑，不依赖 WebGL，便于 Node 自测）
 *
 * 每批 draw call 中 uBoneMatrices 占用 maxBonesPerBatch * 4 个 vertex uniform 向量
 * （一个 mat4 = 4 个 vec4），其余 uniform 占用 BASE_VERTEX_UNIFORM_VECTORS 个。
 */

export const BASE_VERTEX_UNIFORM_VECTORS = 16; // MVP、viewProj、模型参数、padding 等预算
export const VECTORS_PER_MAT4 = 4;
export const BONE_LIMIT_SAFETY_FACTOR = 0.9; // 保守起见只用到上限的 90%

export interface UniformBudgetInfo {
  /** gl.getParameter(MAX_VERTEX_UNIFORM_VECTORS) 的原始值 */
  maxVertexUniformVectors: number;
  /** 扣除基础 uniform 后可用于骨骼矩阵的向量数 */
  availableForBones: number;
  /** 单次 draw call 最多可上传的骨骼矩阵数 */
  maxBonesPerBatch: number;
  /** 是否需要分批 */
  needsBatching: boolean;
}

export interface BatchPlan {
  maxBonesPerBatch: number;
  /** 每批包含的骨骼下标 */
  boneBatches: number[][];
  /** 每批对应的顶点区间 [start, end) 与区间内用到的局部骨骼 id */
  vertexBatches: { start: number; end: number; localBoneIds: number[] }[];
  batchCount: number;
}

/** 由 GPU 上限计算骨骼 uniform 预算 */
export function computeUniformBudget(
  maxVertexUniformVectors: number,
  totalBoneCount: number,
  forcedLimit?: number,
): UniformBudgetInfo {
  const limit = forcedLimit ?? maxVertexUniformVectors;
  if (!Number.isInteger(limit) || limit <= 0) {
    throw new Error(`MAX_VERTEX_UNIFORM_VECTORS 非法：${String(limit)}`);
  }

  const safeLimit = Math.floor(limit * BONE_LIMIT_SAFETY_FACTOR);
  const availableForBones = Math.max(0, safeLimit - BASE_VERTEX_UNIFORM_VECTORS);
  const maxBonesPerBatch = Math.max(1, Math.floor(availableForBones / VECTORS_PER_MAT4));
  return {
    maxVertexUniformVectors: limit,
    availableForBones,
    maxBonesPerBatch,
    needsBatching: totalBoneCount > maxBonesPerBatch,
  };
}

/**
 * 贪心分批：按顶点的全局骨骼 id 收集，能放进当前批次的顶点归一批，
 * 否则关闭当前批再开新批。保证单批骨骼数 <= maxBonesPerBatch。
 */
export function planBatches(
  boneIndices: Uint16Array | Uint8Array | number[],
  bonesPerVertex: number,
  totalBoneCount: number,
  maxBonesPerBatch: number,
): BatchPlan {
  const vertexCount = boneIndices.length / bonesPerVertex;
  if (!Number.isInteger(vertexCount)) {
    throw new Error('boneIndices 长度必须是 bonesPerVertex 的整数倍');
  }

  const boneBatches: number[][] = [];
  const vertexBatches: BatchPlan['vertexBatches'] = [];

  let currentBones = new Set<number>();
  let batchStart = 0;

  const flush = (endVertex: number): void => {
    const globalIds = [...currentBones].sort((a, b) => a - b);
    boneBatches.push(globalIds);
    const localIdOf = new Map<number, number>();
    globalIds.forEach((g, local) => localIdOf.set(g, local));
    const localBoneIds: number[] = [];
    for (let v = batchStart; v < endVertex; v++) {
      for (let k = 0; k < bonesPerVertex; k++) {
        const g = boneIndices[v * bonesPerVertex + k];
        localBoneIds.push(localIdOf.get(g)!);
      }
    }
    vertexBatches.push({ start: batchStart, end: endVertex, localBoneIds });
  };

  for (let v = 0; v < vertexCount; v++) {
    const wanted = new Set<number>();
    for (let k = 0; k < bonesPerVertex; k++) {
      const id = boneIndices[v * bonesPerVertex + k];
      if (id < 0 || id >= totalBoneCount) {
        throw new Error(`顶点 ${v} 引用了越界骨骼 id=${id}（合法范围 0..${totalBoneCount - 1}）`);
      }
      wanted.add(id);
    }

    let fits = true;
    for (const id of wanted) {
      if (!currentBones.has(id) && currentBones.size >= maxBonesPerBatch) {
        fits = false;
        break;
      }
    }

    if (!fits) {
      flush(v);
      currentBones = new Set<number>();
      batchStart = v;
      for (const id of wanted) currentBones.add(id);
    } else {
      for (const id of wanted) currentBones.add(id);
    }
  }
  flush(vertexCount);

  return { maxBonesPerBatch, boneBatches, vertexBatches, batchCount: boneBatches.length };
}
