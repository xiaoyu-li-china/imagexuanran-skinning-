/**
 * skeleton.ts —— JSON 骨骼层级树解析、数据校验与正向运动学（FK）
 *
 * JSON 格式（列主序 4x4 矩阵，长度 16 的 number 数组，与 WebGL 一致）：
 * {
 *   "nodes": [
 *     { "name": "root",  "parent": -1, "localMatrix": [16 个数] },
 *     { "name": "spine", "parent": 0,  "localMatrix": [16 个数] }
 *   ]
 * }
 */
import { Mat4, mat4Identity, mat4Multiply, mat4Invert, mat4Copy } from './mat4.js';

/** JSON 中单个节点的原始数据 */
export interface SkeletonNodeData {
  name: string;
  /** 父节点在 nodes 数组中的下标，根节点为 -1 */
  parent: number;
  /** 列主序局部变换矩阵 */
  localMatrix: number[];
}

export interface SkeletonData {
  nodes: SkeletonNodeData[];
}

export class SkeletonParseError extends Error {
  constructor(message: string) {
    super(`[骨骼数据错误] ${message}`);
    this.name = 'SkeletonParseError';
  }
}

export interface SkeletonNode {
  name: string;
  parent: number;
  local: Mat4;
}

export class Skeleton {
  readonly nodes: SkeletonNode[];
  readonly nameToIndex: Map<string, number>;
  readonly rootIndex: number;
  /** 父节点先于子节点的遍历顺序（FK 计算顺序） */
  readonly traverseOrder: number[];
  /** 绑定姿态（bind pose）下的逆全局矩阵，用于 GPU 蒙皮 */
  readonly inverseBind: Mat4[];
  private readonly globalMatrices: Mat4[];

  private constructor(nodes: SkeletonNode[], rootIndex: number, traverseOrder: number[]) {
    this.nodes = nodes;
    this.rootIndex = rootIndex;
    this.traverseOrder = traverseOrder;
    this.nameToIndex = new Map<string, number>();
    nodes.forEach((n, i) => this.nameToIndex.set(n.name, i));

    // 以 JSON 给出的局部矩阵作为绑定姿态，计算 FK 与逆绑定矩阵
    this.globalMatrices = this.computeFK(nodes.map((n) => n.local));
    this.inverseBind = this.globalMatrices.map((g) => mat4Invert(g));
  }

  get boneCount(): number {
    return this.nodes.length;
  }

  getBoneIndex(name: string): number {
    const idx = this.nameToIndex.get(name);
    if (idx === undefined) {
      throw new SkeletonParseError(`找不到名为 "${name}" 的骨骼节点`);
    }
    return idx;
  }

  /**
   * 正向运动学：global[i] = global[parent(i)] * local[i]
   * @param locals 每个节点的局部矩阵（动画采样结果），缺省用绑定姿态局部矩阵
   * @returns 每个节点的全局矩阵（新数组，不修改内部状态）
   */
  computeFK(locals?: Mat4[]): Mat4[] {
    const src = locals ?? this.nodes.map((n) => n.local);
    if (src.length !== this.nodes.length) {
      throw new Error(`computeFK 失败：局部矩阵数量 ${src.length} 与骨骼数量 ${this.nodes.length} 不一致`);
    }
    const globals = new Array<Mat4>(this.nodes.length);
    for (const i of this.traverseOrder) {
      const p = this.nodes[i].parent;
      globals[i] = p < 0 ? mat4Copy(src[i]) : mat4Multiply(globals[p], src[i]);
    }
    return globals;
  }

  /** 最近一次 computeFK（构造时为绑定姿态）的全局矩阵 */
  getBindGlobal(index: number): Mat4 {
    return mat4Copy(this.globalMatrices[index]);
  }

  /**
   * 解析并校验 JSON 骨骼数据。
   * 任何非法数据都抛出 SkeletonParseError，错误信息包含具体节点与原因。
   */
  static parse(input: string | SkeletonData): Skeleton {
    let data: SkeletonData;
    if (typeof input === 'string') {
      try {
        data = JSON.parse(input) as SkeletonData;
      } catch (e) {
        throw new SkeletonParseError(`JSON 解析失败：${(e as Error).message}`);
      }
    } else {
      data = input;
    }

    if (data === null || typeof data !== 'object' || !Array.isArray(data.nodes)) {
      throw new SkeletonParseError('顶层必须是包含 nodes 数组的对象');
    }
    const rawNodes = data.nodes;
    if (rawNodes.length === 0) {
      throw new SkeletonParseError('nodes 数组为空，至少需要一个根节点');
    }

    const parsed: SkeletonNode[] = [];
    const nameCount = new Map<string, number>();
    const roots: number[] = [];

    for (let i = 0; i < rawNodes.length; i++) {
      const node = rawNodes[i] as Partial<SkeletonNodeData>;
      const label = node && typeof node.name === 'string' ? `"${node.name}"` : `下标 ${i}`;

      if (node === null || typeof node !== 'object') {
        throw new SkeletonParseError(`节点 ${label} 不是对象`);
      }
      if (typeof node.name !== 'string' || node.name.trim() === '') {
        throw new SkeletonParseError(`节点下标 ${i} 的 name 必须是非空字符串`);
      }
      nameCount.set(node.name, (nameCount.get(node.name) ?? 0) + 1);

      if (typeof node.parent !== 'number' || !Number.isInteger(node.parent)) {
        throw new SkeletonParseError(`节点 ${label} 的 parent 必须是整数（根节点为 -1），实际为 ${String(node.parent)}`);
      }
      const parent: number = node.parent;
      if (parent < -1 || parent >= rawNodes.length) {
        throw new SkeletonParseError(
          `节点 ${label} 的 parent=${parent} 越界，合法范围为 -1 到 ${rawNodes.length - 1}`,
        );
      }
      if (parent === i) {
        throw new SkeletonParseError(`节点 ${label} 的 parent 指向自身，构成自引用循环`);
      }

      if (!Array.isArray(node.localMatrix) || node.localMatrix.length !== 16) {
        throw new SkeletonParseError(`节点 ${label} 的 localMatrix 必须是长度为 16 的数组`);
      }
      if (node.localMatrix.some((v) => typeof v !== 'number' || !Number.isFinite(v))) {
        throw new SkeletonParseError(`节点 ${label} 的 localMatrix 包含非有限数值（NaN/Infinity）`);
      }
      const local = new Float32Array(node.localMatrix) as Mat4;
      if (Math.abs(local[15] - 1) > 1e-6 || local[3] !== 0 || local[7] !== 0 || local[11] !== 0) {
        throw new SkeletonParseError(
          `节点 ${label} 的 localMatrix 不是合法仿射矩阵（第 4 行必须为 [0,0,0,1]）`,
        );
      }

      parsed.push({ name: node.name, parent, local });
      if (parent === -1) roots.push(i);
    }

    // 重名校验：列出全部重复项
    const duplicates = [...nameCount.entries()].filter(([, c]) => c > 1).map(([n]) => `"${n}"`);
    if (duplicates.length > 0) {
      throw new SkeletonParseError(`存在重名节点：${duplicates.join('、')}；骨骼名称必须唯一`);
    }

    // 根节点校验：层级树必须有且仅有一个根
    if (roots.length === 0) {
      throw new SkeletonParseError('没有 parent=-1 的根节点，所有节点都挂在某个父节点下（可能存在循环）');
    }
    if (roots.length > 1) {
      throw new SkeletonParseError(
        `存在 ${roots.length} 个根节点（下标 ${roots.join(', ')}），骨骼层级树只能有一个根`,
      );
    }

    // 循环引用 / 不可达校验：三色 DFS
    const order = Skeleton.topoOrder(parsed, roots[0]);

    return new Skeleton(parsed, roots[0], order);
  }

  /** DFS 生成父先子后的遍历顺序；检测环并给出环路上的节点名 */
  private static topoOrder(nodes: SkeletonNode[], root: number): number[] {
    const color = new Uint8Array(nodes.length); // 0=白 1=灰(在当前路径) 2=黑(完成)
    const order: number[] = [];

    const children: number[][] = nodes.map(() => []);
    nodes.forEach((n, i) => {
      if (n.parent >= 0) children[n.parent].push(i);
    });

    const walk = (i: number, path: number[]): void => {
      if (color[i] === 1) {
        const cycleStart = path.indexOf(i);
        const cycleNames = path
          .slice(cycleStart)
          .concat(i)
          .map((idx) => `"${nodes[idx].name}"`)
          .join(' -> ');
        throw new SkeletonParseError(`检测到循环引用，环路为：${cycleNames}`);
      }
      if (color[i] === 2) return;
      color[i] = 1;
      path.push(i);
      for (const c of children[i]) walk(c, path);
      path.pop();
      color[i] = 2;
      order.push(i);
    };

    walk(root, []);

    if (order.length !== nodes.length) {
      const unreachable = nodes
        .map((_, i) => i)
        .filter((i) => color[i] !== 2)
        .map((i) => `"${nodes[i].name}"`)
        .join('、');
      throw new SkeletonParseError(`以下节点无法从根节点到达（悬挂的循环子树）：${unreachable}`);
    }

    // DFS 后序是子先父后，FK 需要父先子后
    order.reverse();
    return order;
  }

  /** 便捷构造：单位矩阵骨架（测试/程序化生成用） */
  static identityLocal(): Mat4 {
    return mat4Identity();
  }
}
