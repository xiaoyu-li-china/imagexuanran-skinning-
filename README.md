# imagexuanran-skinning —— 原生 WebGL2 骨骼动画基础管线

仅使用 **原生 WebGL2 + GLSL ES 3.00 + TypeScript + Vite** 实现的骨骼动画最小管线。
**不依赖任何 3D 引擎或现成动画库**（无 Three.js / Babylon.js / glTF 运行时等），数学库、
动画采样、状态机、分批规划全部手写；不使用 Docker。本阶段只包含骨骼动画基础管线，
**不含阴影、不含后期处理**。

## 快速开始

```bash
npm install
npm run dev          # 开发服务器，浏览器打开 http://localhost:5173

npm run build        # tsc 严格类型检查 + vite 生产构建（输出 dist/）
npm run selftest     # Node 下运行 34 项自测（无需浏览器）
```

浏览器内验证（需要本机装有 Chrome）：

```bash
npm run build
npm run preview -- --port 5179          # 终端 A
node scripts/cdp-test.mjs               # 终端 B：无头 Chrome 自动化验证
```

页面 URL 参数：

- `?selftest=1`：打开页面后自动在控制台运行自测；
- `?forcelimit=1`：把 vertex uniform 预算强制压到 32 vectors，演示分批降级。

## 目录结构

```
src/
  core/
    mat4.ts        列主序 mat4 / 四元数数学库（乘法、求逆、行列式、TRS 分解/重组、slerp）
    skeleton.ts    JSON 骨骼树解析、数据校验、正向运动学（FK）、逆绑定矩阵
    animation.ts   动画片段解析、关键帧采样（lerp/slerp）、带 Crossfade 的状态机
    batching.ts    uniform 预算计算与分批规划（纯逻辑，Node 可单测）
    camera.ts      透视投影 / lookAt
  webgl/
    renderer.ts    WebGL2 渲染器：uBoneMatrices 注入、LBS 蒙皮 shader、分批 draw call
    meshBuilder.ts 程序化蒙皮网格（4 段方柱，每顶点 2 骨骼权重）
  assets/
    skeleton.json      5 节点骨骼链（由 scripts/generate-assets.mjs 生成）
    animations.json    idle / walk 两个片段（旋转 + 缩放关键帧）
  selftest/
    runSelftest.ts 34 项自测（浏览器与 Node 共用同一套代码）
  main.ts          页面入口：加载资产、UI、渲染循环
scripts/
  generate-assets.mjs  资产生成器
  cdp-test.mjs         零依赖 Chrome DevTools Protocol 无头验证
```

## 一、骨骼 JSON 格式与矩阵层级计算（FK）

### JSON 格式

```json
{
  "nodes": [
    { "name": "root",  "parent": -1, "localMatrix": [/* 16 个列主序数 */] },
    { "name": "bone1", "parent": 0,  "localMatrix": [/* ... */] }
  ]
}
```

- 所有矩阵为 **列主序 4×4**（数组下标 `m[col*4+row]`，平移在 `m[12..14]`），与 WebGL
  uniform 内存布局一致，可直接 `uniformMatrix4fv` 上传；
- `parent` 为父节点在 `nodes` 中的下标，根节点为 `-1`。

### 正向运动学（FK）

解析后先做一次 DFS 得到**父先子后**的遍历顺序 `traverseOrder`，每帧按该顺序计算：

```
global[root] = local[root]
global[i]    = global[parent(i)] × local[i]
```

因为父节点保证先于子节点被计算，单次线性遍历即可得到全部全局矩阵，复杂度 O(n)。

构造骨架时，把 JSON 给出的局部矩阵视为**绑定姿态（bind pose）**，并预计算每根骨骼的
逆绑定矩阵：

```
inverseBind[i] = inverse(global_bind[i])
```

### 非法数据校验（抛出带节点名的可读错误 `SkeletonParseError`）

| 非法情况 | 处理 |
| --- | --- |
| JSON 语法错误 | 包装为 `[骨骼数据错误] JSON 解析失败：…` |
| 节点重名 | 列出全部重名节点并拒绝 |
| `parent` 越界 / 非整数 / 指向自身 | 指出节点名与非法值 |
| 多个根节点 / 无根节点 | 报告根下标列表 |
| 循环引用 | 三色 DFS：可达节点成环必然与根断开，报“无法从根节点到达”的节点清单；灰节点再次命中时报告具体环路 |
| 矩阵长度 ≠ 16、含 NaN/Infinity、第 4 行不是 `[0,0,0,1]` | 拒绝非仿射/非有限矩阵 |

## 二、动画片段、状态机与 Crossfade 原理

### 片段格式与采样

每个片段（`AnimationClip`）包含若干轨道，每根骨骼至多一条轨道；关键帧支持 TRS
（translation / rotation 四元数 / scale）或直接给 4×4 矩阵（加载时用 `mat4Decompose`
统一分解为 TRS，分解会检测零缩放轴并对反射做轴翻转）。

采样时时间按片段时长取模循环，二分查找相邻关键帧，通道分别插值：

- 平移、缩放：线性插值 `lerp`；
- 旋转：四元数球面线性插值 `slerp`，自动取最短弧，夹角很小时退化为归一化线性插值。

### 状态机与 Crossfade（秒级时间驱动，无瞬间跳变）

`AnimationStateMachine` 同时维护若干**活动源**，每个源有自己的片段本地时间与权重。
调用 `play(name, fadeSeconds)` 切换时：

1. 现有全部源转入**淡出**，记录淡出起始权重 `fadeStart`；
2. 新源以**权重 0 淡入**——因此切换当帧输出姿态与切换前完全一致（自测实测最大矩阵
   元素差为 `0.00e+0`），不存在姿态跳变；
3. 之后每帧用真实秒数 `dt` 推进淡入淡出进度 `p = elapsed / fadeSeconds`，权重曲线用
   smoothstep `3p²−2p³` 缓动（端点导数为 0，切换起止处速度连续）：

```
w_in(t)  = smoothstep(t / fade)                       // 0 → 1
w_out(t) = fadeStart × (1 − smoothstep(t / fade))     // fadeStart → 0
```

4. 每帧对所有源的权重**归一化**后混合：平移/缩放按权重线性混合，旋转用累积 slerp
   （在已累积姿态与下一姿态之间按相对权重插值），再把混合后的 TRS 重组为局部矩阵，
   交给 FK 计算全局矩阵。淡出完成的源被回收，稳态下只剩一个权重为 1 的片段。

关键帧插值与跨片段混合都在 TRS 空间进行，避免了直接对矩阵插值产生的非刚性变形。

## 三、GPU 蒙皮与 uniform 数组注入

### 蒙皮矩阵约定

每帧 CPU 计算当前姿态全局矩阵后，构造标准 offset 矩阵并以 `Float32Array` 整体上传：

```
offsetBone[i] = global_current[i] × inverseBind[i]
gl.uniformMatrix4fv(uBoneMatrices[0], false, offsetBones)
```

顶点着色器（GLSL ES 3.00）做每顶点 2 骨骼线性混合蒙皮（LBS）：

```glsl
mat4 skin = uBoneMatrices[aBoneIndices.x] * aWeights.x
          + uBoneMatrices[aBoneIndices.y] * aWeights.y;
vec4 skinnedPos = skin * vec4(aPosition, 1.0);
```

整数骨骼下标用 `vertexAttribIPointer` + `uvec2` 传入，权重和在片段着色器中做可视化校验
（偏离 1 时顶点偏红）。

### MAX_VERTEX_UNIFORM_VECTORS 校验与分批降级

一个 `mat4` 占 **4 个 vertex uniform 向量**。渲染器构造时读取
`gl.getParameter(MAX_VERTEX_UNIFORM_VECTORS)`，按以下公式预算：

```
安全预算   = floor(limit × 0.9)
骨骼可用   = 安全预算 − 16（uViewProj/uModel/uNormalMat 等基础 uniform 预留）
单批上限   = floor(骨骼可用 / 4)
```

- 骨骼数 ≤ 单批上限：一个 draw call 传全部骨骼矩阵；
- 超出上限：`batching.ts` 按顶点顺序贪心分批——扫描每个顶点引用的骨骼，当前批次
  容量装不下就关闭当前批、开新批；每批把全局骨骼 id 重映射为批次内局部 id（shader
  数组下标），重建该批的 VAO/索引缓冲，逐批 draw。**渲染不中断**，同时通过
  `console.warn` 与页面信息面板输出可读警告。
- 页面勾选“强制 uniform 向量预算”（或 `?forcelimit=1`）可把预算压到 32 vectors 模拟
  低端 GPU：本 demo 的 5 根骨骼会被拆成 2 个 draw call（无头浏览器实测）。

### 分批渲染的限制

1. **draw call 上升**：批次数随骨骼数/顶点分布增长，最坏情况下接近
   `ceil(骨骼数 / 单批上限)`；
2. **跨批顶点无法共享影响骨骼**：一个顶点的全部影响骨骼必须落在同一批，分批只按
   顶点切分、不拆分单个顶点的权重，因此网格需要按骨骼引用做局部性排序（本 demo 的
   程序化网格天然分段连续）；
3. 每批的 shader 数组容量按预算固定（`uniform mat4 uBoneMatrices[MAX_BONES]` 需要
   编译期常量），批次实际骨骼数可能小于容量，存在少量 uniform 空间浪费；
4. 每批都要重设全部公共 uniform 并产生一次绘制调用，状态切换开销随批数线性增加。

## 四、自测说明

`npm run selftest`（Node）或页面按钮 / `?selftest=1`（浏览器控制台）运行 **34 项**断言：

- **A 组（4 项）**：骨骼解析、FK 绑定姿态全局坐标的解析解验证；
- **B 组（10 项）**：重名、循环、自引用、父节点越界、多根、坏矩阵、坏 JSON 等全部
  抛出可读错误；
- **C 组（11 项）**：idle/walk 加载、切换瞬间零跳变、Crossfade 中点权重 0.5/0.5、
  中间插值帧 bone2/bone4 全局坐标输出，并与**独立编写的参考采样/混合实现**（直接读
  JSON 关键帧，不复用生产采样器）逐通道对比；
- **D 组（2 项）**：60 帧 crossfade 过程中所有局部/全局/offset 矩阵行列式非奇异；
- **E 组（6 项）**：uniform 预算公式、强制低预算触发分批、分批覆盖完整性与局部 id
  合法性；
- **F 组（4 项）**：TRS 分解/重组往返误差、slerp/lerp 端点性质。

## 五、验证结果（2026-11-12 实测）

| 验证手段 | 结果 |
| --- | --- |
| `tsc --noEmit` 严格模式 | 通过，零错误 |
| `vite build` 生产构建 | 通过，无外部化/打包警告 |
| Node `npm run selftest` | **34/34 通过** |
| 无头 Chrome（CDP）浏览器内自测 | **34/34 通过，无未捕获异常** |
| WebGL2 上下文 / 真实 GPU 上限读取 | 通过（实测 `MAX_VERTEX_UNIFORM_VECTORS=1024`） |
| `?forcelimit=1` 分批降级 | 单批上限 3 根、5 根骨骼拆为 2 个 draw call、警告正常输出、渲染不中断 |
| idle→walk crossfade | 中点实测权重 0.500/0.500，结束后仅剩 walk |
| 渲染截图 | `scripts/screenshots/`（默认、分批、walk 极值帧均可见多关节蒙皮变形） |
