/**
 * renderer.ts —— 原生 WebGL2 GPU 蒙皮渲染器
 *
 * 职责：
 *  1. 校验 MAX_VERTEX_UNIFORM_VECTORS，计算 uBoneMatrices 预算；
 *  2. 超限时按 batching.ts 的规划分批渲染并通过 console.warn 给出警告，不崩溃；
 *  3. 每批把骨骼矩阵（offsetBone = currentGlobal * inverseBind）注入 uniform 数组；
 *  4. 顶点着色器做每顶点 2 骨骼线性混合蒙皮（LBS），本阶段不含阴影与后期处理。
 *
 * 批次的 VAO / 顶点缓冲 / 索引缓冲在 setMesh 时一次性建立，渲染帧只更新
 * uBoneMatrices，避免每帧分配 GL 资源。
 */
import { Mat4, mat4Multiply } from '../core/mat4.js';
import { Skeleton } from '../core/skeleton.js';
import {
  BatchPlan,
  UniformBudgetInfo,
  computeUniformBudget,
  planBatches,
} from '../core/batching.js';

export interface SkinnedMeshData {
  positions: Float32Array; // vec3
  normals: Float32Array; // vec3
  boneIndices: Uint16Array; // 每顶点 bonesPerVertex 个全局骨骼 id
  weights: Float32Array; // 每顶点 bonesPerVertex 个权重，和为 1
  bonesPerVertex: number; // 本渲染器固定支持 2
  indexCount: number;
  indices: Uint16Array;
}

const MAX_INFLUENCES = 2;

const VERT_SOURCE = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in uvec2 aBoneIndices;
layout(location = 3) in vec2 aWeights;

uniform mat4 uViewProj;
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uBoneMatrices[MAX_BONES];

out vec3 vNormal;
out vec3 vWorldPos;
out float vWeightSum;

void main() {
  // 线性混合蒙皮：skinned = Σ weight_i * offsetBone_i * pos
  mat4 skin =
      uBoneMatrices[aBoneIndices.x] * aWeights.x
    + uBoneMatrices[aBoneIndices.y] * aWeights.y;

  vec4 skinnedPos = skin * vec4(aPosition, 1.0);
  vec3 skinnedNormal = mat3(skin) * aNormal;
  vWeightSum = aWeights.x + aWeights.y;

  vec4 world = uModel * skinnedPos;
  vWorldPos = world.xyz;
  vNormal = normalize(uNormalMat * skinnedNormal);
  gl_Position = uViewProj * world;
}
`;

const FRAG_SOURCE = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vWorldPos;
in float vWeightSum;
uniform vec3 uBaseColor;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  vec3 lightDir = normalize(vec3(0.5, 0.9, 0.7));
  float lambert = max(dot(n, lightDir), 0.0);
  vec3 color = uBaseColor * (0.28 + 0.72 * lambert);
  // 权重和偏离 1 时偏红，作为蒙皮数据异常的可视化提示
  float bad = clamp(abs(vWeightSum - 1.0) * 4.0, 0.0, 1.0);
  color = mix(color, vec3(1.0, 0.2, 0.2), bad);
  outColor = vec4(color, 1.0);
}
`;

interface BatchGpuResources {
  program: WebGLProgram;
  vao: WebGLVertexArrayObject;
  buffers: WebGLBuffer[];
  indexBuffer: WebGLBuffer;
  indexCount: number;
  capacity: number;
  globalBoneIds: number[];
  uniforms: {
    viewProj: WebGLUniformLocation | null;
    model: WebGLUniformLocation | null;
    normalMat: WebGLUniformLocation | null;
    baseColor: WebGLUniformLocation | null;
    boneMatrices: WebGLUniformLocation | null;
  };
}

export class RendererError extends Error {
  constructor(message: string) {
    super(`[WebGL2 渲染错误] ${message}`);
    this.name = 'RendererError';
  }
}

export class SkinningRenderer {
  private gl: WebGL2RenderingContext;
  private skeleton: Skeleton;
  private budget: UniformBudgetInfo;
  private batchPlan: BatchPlan | null = null;
  private batchResources: BatchGpuResources[] = [];

  readonly warnings: string[] = [];

  constructor(canvas: HTMLCanvasElement, skeleton: Skeleton, forcedVertexUniformLimit?: number) {
    const gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) {
      throw new RendererError('当前环境不支持 WebGL2，无法创建上下文');
    }
    this.gl = gl;
    this.skeleton = skeleton;

    const rawLimit = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) as number;
    this.budget = computeUniformBudget(rawLimit, skeleton.boneCount, forcedVertexUniformLimit);

    if (this.budget.needsBatching) {
      const msg =
        `骨骼数量 ${skeleton.boneCount} 超过单批 uniform 容量 ` +
        `（MAX_VERTEX_UNIFORM_VECTORS=${this.budget.maxVertexUniformVectors}，` +
        `扣除基础 uniform 后单批最多 ${this.budget.maxBonesPerBatch} 根骨骼），` +
        `已自动降级为分批渲染，draw call 增加`;
      this.warnings.push(msg);
      console.warn('[SkinningRenderer] ⚠️ ' + msg);
    } else {
      console.info(
        `[SkinningRenderer] MAX_VERTEX_UNIFORM_VECTORS=${this.budget.maxVertexUniformVectors}，` +
          `单批可容纳 ${this.budget.maxBonesPerBatch} 根骨骼，当前 ${skeleton.boneCount} 根无需分批`,
      );
    }

    gl.enable(gl.DEPTH_TEST);
  }

  getBudgetInfo(): UniformBudgetInfo {
    return { ...this.budget };
  }

  getBatchPlan(): BatchPlan | null {
    return this.batchPlan;
  }

  get drawCallCount(): number {
    return this.batchResources.length;
  }

  setMesh(mesh: SkinnedMeshData): void {
    const { boneIndices, weights, bonesPerVertex, positions, normals, indexCount } = mesh;
    if (bonesPerVertex !== MAX_INFLUENCES) {
      throw new RendererError(`本渲染器固定支持每顶点 ${MAX_INFLUENCES} 骨骼，实际为 ${bonesPerVertex}`);
    }
    if (positions.length !== normals.length) {
      throw new RendererError('positions 与 normals 顶点数不一致');
    }
    if (boneIndices.length !== weights.length) {
      throw new RendererError('boneIndices 与 weights 长度不一致');
    }
    const vertexCount = positions.length / 3;
    if (boneIndices.length !== vertexCount * bonesPerVertex) {
      throw new RendererError('每顶点骨骼属性数量不匹配');
    }
    for (let v = 0; v < vertexCount; v++) {
      let sum = 0;
      for (let k = 0; k < bonesPerVertex; k++) sum += weights[v * bonesPerVertex + k];
      if (Math.abs(sum - 1) > 1e-4) {
        throw new RendererError(`顶点 ${v} 的蒙皮权重和为 ${sum}，必须归一化为 1`);
      }
    }
    if (indexCount % 3 !== 0) {
      throw new RendererError('索引数必须是 3 的倍数（三角形列表）');
    }

    this.batchPlan = planBatches(
      boneIndices,
      bonesPerVertex,
      this.skeleton.boneCount,
      this.budget.maxBonesPerBatch,
    );
    if (this.batchPlan.batchCount > 1) {
      const msg = `网格被拆分为 ${this.batchPlan.batchCount} 个 draw call 进行分批蒙皮渲染`;
      if (!this.warnings.includes(msg)) {
        this.warnings.push(msg);
        console.warn('[SkinningRenderer] ⚠️ ' + msg);
      }
    }

    this.disposeBatches();
    this.batchPlan.vertexBatches.forEach((vb, batchIdx) => {
      const globalBoneIds = this.batchPlan!.boneBatches[batchIdx];
      // 统一用预算容量作为 shader 数组大小，program 可跨批次复用
      const capacity = this.budget.maxBonesPerBatch;
      this.batchResources.push(this.buildBatch(mesh, vb, globalBoneIds, capacity));
    });
  }

  /** 为一个批次预建 shader/VAO/缓冲，之后每帧只更新 uniform */
  private buildBatch(
    mesh: SkinnedMeshData,
    vb: { start: number; end: number; localBoneIds: number[] },
    globalBoneIds: number[],
    capacity: number,
  ): BatchGpuResources {
    const gl = this.gl;
    const vertSrc = VERT_SOURCE.replace('MAX_BONES', String(capacity));
    const program = this.linkProgram(vertSrc, FRAG_SOURCE);

    const bpv = mesh.bonesPerVertex;
    const vertexCount = vb.end - vb.start;
    const positions = mesh.positions.subarray(vb.start * 3, vb.end * 3);
    const normals = mesh.normals.subarray(vb.start * 3, vb.end * 3);
    const localIndicesU16 = new Uint16Array(vb.localBoneIds);
    const batchWeights = new Float32Array(vertexCount * bpv);
    for (let v = 0; v < vertexCount; v++) {
      for (let k = 0; k < bpv; k++) {
        batchWeights[v * bpv + k] = mesh.weights[(vb.start + v) * bpv + k];
      }
    }

    // 重映射索引到本批局部顶点
    const indexMap = new Map<number, number>();
    const localIndices: number[] = [];
    for (let i = 0; i < mesh.indexCount; i++) {
      const g = mesh.indices[i];
      if (g < vb.start || g >= vb.end) continue;
      let local = indexMap.get(g);
      if (local === undefined) {
        local = indexMap.size;
        indexMap.set(g, local);
      }
      localIndices.push(local);
    }
    const indexArray = new Uint16Array(localIndices);

    const vao = gl.createVertexArray();
    if (!vao) throw new RendererError('创建 VAO 失败');
    gl.bindVertexArray(vao);
    const buffers: WebGLBuffer[] = [];
    buffers.push(this.uploadAttribute(0, positions, 3, gl.FLOAT, false));
    buffers.push(this.uploadAttribute(1, normals, 3, gl.FLOAT, false));
    buffers.push(this.uploadAttribute(2, localIndicesU16, 2, gl.UNSIGNED_SHORT, true));
    buffers.push(this.uploadAttribute(3, batchWeights, 2, gl.FLOAT, false));

    const indexBuffer = gl.createBuffer();
    if (!indexBuffer) throw new RendererError('创建索引 buffer 失败');
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexArray, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    return {
      program,
      vao,
      buffers,
      indexBuffer,
      indexCount: indexArray.length,
      capacity,
      globalBoneIds: [...globalBoneIds],
      uniforms: {
        viewProj: gl.getUniformLocation(program, 'uViewProj'),
        model: gl.getUniformLocation(program, 'uModel'),
        normalMat: gl.getUniformLocation(program, 'uNormalMat'),
        baseColor: gl.getUniformLocation(program, 'uBaseColor'),
        boneMatrices: gl.getUniformLocation(program, 'uBoneMatrices[0]'),
      },
    };
  }

  private uploadAttribute(
    location: number,
    data: ArrayBufferView,
    components: number,
    type: number,
    integer: boolean,
  ): WebGLBuffer {
    const gl = this.gl;
    const buf = gl.createBuffer();
    if (!buf) throw new RendererError('创建顶点 buffer 失败');
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(location);
    if (integer) {
      gl.vertexAttribIPointer(location, components, type, 0, 0);
    } else {
      gl.vertexAttribPointer(location, components, type, false, 0, 0);
    }
    return buf;
  }

  private linkProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string, label: string): WebGLShader => {
      const shader = gl.createShader(type);
      if (!shader) throw new RendererError(`创建 ${label} shader 失败`);
      gl.shaderSource(shader, src);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new RendererError(`${label} shader 编译失败：${log}\n--- 源码 ---\n${src}`);
      }
      return shader;
    };
    const vs = compile(gl.VERTEX_SHADER, vertSrc, '顶点');
    const fs = compile(gl.FRAGMENT_SHADER, fragSrc, '片段');
    const program = gl.createProgram();
    if (!program) throw new RendererError('创建 program 失败');
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new RendererError(`program 链接失败：${log}`);
    }
    return program;
  }

  /**
   * 渲染一帧。
   * @param globalMatrices 当前姿态下每根骨骼的全局矩阵
   */
  render(globalMatrices: Mat4[], viewProj: Mat4, model: Mat4, baseColor: [number, number, number]): void {
    if (this.batchResources.length === 0) {
      throw new RendererError('尚未调用 setMesh 设置蒙皮网格');
    }
    const gl = this.gl;

    // 标准 GPU 蒙皮约定：offsetBone = currentGlobal * inverseBind
    const offsetBones = globalMatrices.map((g, i) => mat4Multiply(g, this.skeleton.inverseBind[i]));

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0.063, 0.078, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const normalMat = new Float32Array([
      model[0], model[1], model[2],
      model[4], model[5], model[6],
      model[8], model[9], model[10],
    ]);

    for (const batch of this.batchResources) {
      const batchMatrices = new Float32Array(batch.capacity * 16);
      batch.globalBoneIds.forEach((globalBone, local) => {
        batchMatrices.set(offsetBones[globalBone], local * 16);
      });

      gl.useProgram(batch.program);
      gl.bindVertexArray(batch.vao);
      gl.uniformMatrix4fv(batch.uniforms.viewProj, false, viewProj);
      gl.uniformMatrix4fv(batch.uniforms.model, false, model);
      gl.uniformMatrix3fv(batch.uniforms.normalMat, false, normalMat);
      gl.uniform3fv(batch.uniforms.baseColor, new Float32Array(baseColor));
      gl.uniformMatrix4fv(batch.uniforms.boneMatrices, false, batchMatrices);
      gl.drawElements(gl.TRIANGLES, batch.indexCount, gl.UNSIGNED_SHORT, 0);
    }
    gl.bindVertexArray(null);
  }

  private disposeBatches(): void {
    const gl = this.gl;
    for (const b of this.batchResources) {
      gl.deleteProgram(b.program);
      gl.deleteVertexArray(b.vao);
      gl.deleteBuffer(b.indexBuffer);
      for (const buf of b.buffers) gl.deleteBuffer(buf);
    }
    this.batchResources = [];
  }

  dispose(): void {
    this.disposeBatches();
  }
}
