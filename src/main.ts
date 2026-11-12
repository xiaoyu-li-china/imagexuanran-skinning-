/**
 * main.ts —— 应用入口：加载资产 -> 构建骨架/状态机/渲染器 -> 渲染循环与 UI
 */
import { Skeleton, SkeletonData } from './core/skeleton.js';
import { AnimationClip, AnimationStateMachine } from './core/animation.js';
import { mat4FromTranslation, Mat4 } from './core/mat4.js';
import { mat4Perspective, mat4LookAt, viewProjection } from './core/camera.js';
import { SkinningRenderer } from './webgl/renderer.js';
import { buildSkinnedLimb } from './webgl/meshBuilder.js';
import { runSelftest } from './selftest/runSelftest.js';
import skeletonJson from './assets/skeleton.json';
import animationsJson from './assets/animations.json';

interface AnimationsFile {
  clips: Parameters<typeof AnimationClip.parse>[0][];
}

async function bootstrap(): Promise<void> {
  const infoEl = document.getElementById('info') as HTMLPreElement;
  const appendInfo = (msg: string): void => {
    infoEl.textContent += `\n${msg}`;
  };
  infoEl.textContent = '初始化中…';

  // 1. 解析骨骼（非法数据会在这里抛出可读错误）
  const skeleton = Skeleton.parse(skeletonJson as SkeletonData);
  appendInfo(`骨骼解析成功：${skeleton.boneCount} 根，根节点 "${skeleton.nodes[skeleton.rootIndex].name}"`);

  // 2. 动画片段与状态机
  const fsm = new AnimationStateMachine(skeleton);
  const animFile = animationsJson as AnimationsFile;
  for (const clipData of animFile.clips) {
    fsm.loadClip(AnimationClip.parse(clipData, skeleton));
  }
  const fadeInput = document.getElementById('fade') as HTMLInputElement;
  const fadeVal = document.getElementById('fade-val') as HTMLSpanElement;
  fsm.setDefaultFade(Number(fadeInput.value));
  fsm.play('idle');
  appendInfo('已加载片段：idle、walk；当前播放 idle');

  // 3. 渲染器（可勾选强制低 uniform 预算，演示分批降级；也可用 ?forcelimit=1 强制）
  const params = new URLSearchParams(window.location.search);
  const forceCheckbox = document.getElementById('force-limit') as HTMLInputElement;
  if (params.get('forcelimit') === '1') forceCheckbox.checked = true;
  const canvas = document.getElementById('gl-canvas') as HTMLCanvasElement;

  let renderer = new SkinningRenderer(
    canvas,
    skeleton,
    forceCheckbox.checked ? 32 : undefined,
  );
  renderer.setMesh(buildSkinnedLimb());
  const budget = renderer.getBudgetInfo();
  appendInfo(
    `MAX_VERTEX_UNIFORM_VECTORS=${budget.maxVertexUniformVectors}，` +
      `单批上限 ${budget.maxBonesPerBatch} 根骨骼，` +
      `实际 draw call=${renderer.drawCallCount}`,
  );
  renderer.warnings.forEach((w) => appendInfo(`⚠️ ${w}`));

  // 4. 相机（aspect 随 canvas 实际尺寸动态更新）
  let proj = mat4Perspective(Math.PI / 4, 1, 0.1, 100);
  const view = mat4LookAt([2.6, 2.6, 5.2], [0, 1.7, 0], [0, 1, 0]);
  let viewProj = viewProjection(proj, view);
  const model: Mat4 = mat4FromTranslation(0, 0, 0);

  const resize = (): void => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      proj = mat4Perspective(Math.PI / 4, w / h, 0.1, 100);
      viewProj = viewProjection(proj, view);
    }
  };
  resize();
  window.addEventListener('resize', resize);

  // 5. UI 交互
  const btnIdle = document.getElementById('btn-idle') as HTMLButtonElement;
  const btnWalk = document.getElementById('btn-walk') as HTMLButtonElement;
  btnIdle.addEventListener('click', () => {
    fsm.play('idle');
    appendInfo(`-> 请求切换 idle（crossfade ${Number(fadeInput.value).toFixed(2)}s）`);
  });
  btnWalk.addEventListener('click', () => {
    fsm.play('walk');
    appendInfo(`-> 请求切换 walk（crossfade ${Number(fadeInput.value).toFixed(2)}s）`);
  });
  fadeInput.addEventListener('input', () => {
    const v = Number(fadeInput.value);
    fadeVal.textContent = v.toFixed(2);
    fsm.setDefaultFade(v);
  });
  forceCheckbox.addEventListener('change', () => {
    // 用新预算重建渲染器，验证降级路径不崩溃
    renderer.dispose();
    renderer = new SkinningRenderer(canvas, skeleton, forceCheckbox.checked ? 32 : undefined);
    renderer.setMesh(buildSkinnedLimb());
    const b = renderer.getBudgetInfo();
    appendInfo(
      `重建渲染器：预算=${b.maxVertexUniformVectors}，单批 ${b.maxBonesPerBatch} 根，` +
        `draw call=${renderer.drawCallCount}`,
    );
    renderer.warnings.forEach((w) => appendInfo(`⚠️ ${w}`));
  });
  (document.getElementById('btn-selftest') as HTMLButtonElement).addEventListener('click', () => {
    void runBrowserSelftest();
  });

  async function runBrowserSelftest(): Promise<void> {
    const result = await runSelftest({
      skeletonData: skeletonJson as SkeletonData,
      animationsData: animationsJson as { clips: Parameters<typeof AnimationClip.parse>[0][] },
    });
    // 带标记的输出便于无头浏览器抓取验证
    for (const line of result.lines) console.log(`SELFTEST_LINE ${line}`);
    console.log(`SELFTEST_DONE pass=${result.passed} fail=${result.failed}`);
    appendInfo(`自测完成：通过 ${result.passed}，失败 ${result.failed}（详见控制台）`);
  }

  // URL 带 ?selftest=1 时自动运行（CI / 无头浏览器验证用）
  if (params.get('selftest') === '1') {
    void runBrowserSelftest();
  }

  // 6. 渲染循环（last 用首帧 rAF 时间戳初始化：rAF 时间戳是帧开始时刻，
  //    可能早于 performance.now()，混用会导致首帧 dt 为负）
  let last: number | null = null;
  let frameCount = 0;
  const loop = (now: number): void => {
    if (last === null) last = now;
    const dt = Math.min(Math.max((now - last) / 1000, 0), 0.1); // 钳制到 [0, 0.1s]
    last = now;

    const locals = fsm.update(dt);
    const globals = skeleton.computeFK(locals);
    renderer.render(globals, viewProj, model, [0.36, 0.62, 0.92]);

    frameCount++;
    if (frameCount % 60 === 0) {
      const snap = fsm.snapshot();
      appendInfo(
        `[${(now / 1000).toFixed(1)}s] 主片段=${snap.primary}，` +
          snap.sources.map((s) => `${s.name} w=${s.weight.toFixed(2)}`).join(' | ') +
          `，draw=${renderer.drawCallCount}`,
      );
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // 暴露给控制台便于手动验证
  (globalThis as Record<string, unknown>).__skinning = { skeleton, fsm, getRenderer: () => renderer };
}

bootstrap().catch((e) => {
  const infoEl = document.getElementById('info');
  if (infoEl) infoEl.textContent = `初始化失败：${e instanceof Error ? e.message : String(e)}`;
  console.error(e);
});
