/**
 * cdp-test.mjs —— 零依赖 Chrome DevTools Protocol 无头验证
 * 真实启动 Chrome（WebGL2 + SwiftShader/ANGLE），验证：
 *   1. 页面正常初始化、WebGL2 上下文与 shader 链接成功
 *   2. 浏览器内自测 33 项全过（抓取 SELFTEST_DONE）
 *   3. ?forcelimit=1 触发分批降级（draw call > 1）且只警告不崩溃
 *   4. 截图保存到 scripts/screenshots/ 供人工查看
 */
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CDP_PORT = 9333;
const APP_BASE = process.env.APP_BASE ?? 'http://localhost:5179';
const shotDir = join(here, 'screenshots');
mkdirSync(shotDir, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const getJson = async (url) => (await fetch(url)).json();

let failures = 0;
const assert = (name, cond, detail = '') => {
  if (cond) console.log(`  ✅ ${name}${detail ? ` —— ${detail}` : ''}`);
  else {
    failures++;
    console.log(`  ❌ ${name}${detail ? `—— ${detail}` : ''}`);
  }
};

async function main() {
  const chrome = spawn(
    CHROME,
    [
      '--headless=new',
      `--remote-debugging-port=${CDP_PORT}`,
      '--use-angle=metal',
      '--enable-unsafe-swiftshader',
      '--disable-gpu-sandbox',
      '--no-first-run',
      '--no-default-browser-check',
      '--user-data-dir=/tmp/chrome-cdp-skinning-profile',
      'about:blank',
    ],
    { stdio: 'ignore' },
  );

  try {
    let targets = null;
    for (let i = 0; i < 60; i++) {
      try {
        targets = await getJson(`http://127.0.0.1:${CDP_PORT}/json`);
        if (targets.length) break;
      } catch {
        /* 等待 DevTools 端口就绪 */
      }
      await sleep(250);
    }
    if (!targets) throw new Error('Chrome DevTools 端口未就绪');

    const page = targets.find((t) => t.type === 'page');
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
    });

    let msgId = 0;
    const pending = new Map();
    const consoleLines = [];
    const logEntries = [];
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id) {
        pending.get(msg.id)?.resolve(msg);
        pending.delete(msg.id);
        return;
      }
      if (msg.method === 'Runtime.consoleAPICalled') {
        const text = (msg.params.args ?? [])
          .map((a) => a.value ?? a.description ?? JSON.stringify(a.preview ?? ''))
          .join(' ');
        consoleLines.push(text);
      }
      if (msg.method === 'Log.entryAdded') logEntries.push(msg.params.entry.text);
      if (msg.method === 'Runtime.exceptionThrown') {
        consoleLines.push('UNCAUGHT ' + JSON.stringify(msg.params.exceptionDetails.exception?.description));
      }
    };
    const send = (method, params = {}) =>
      new Promise((resolve, reject) => {
        const id = ++msgId;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
      });
    await send('Runtime.enable');
    await send('Log.enable');
    await send('Page.enable');

    const evaluate = async (expression) => {
      const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
      if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.text);
      return r.result.result.value;
    };
    const navigate = async (url) => {
      consoleLines.length = 0;
      logEntries.length = 0;
      await send('Page.navigate', { url });
    };
    const waitForLine = async (needle, timeoutMs = 20000) => {
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        if (consoleLines.some((l) => l.includes(needle))) return true;
        await sleep(200);
      }
      return false;
    };
    const screenshot = async (name) => {
      const r = await send('Page.captureScreenshot', { format: 'png' });
      writeFileSync(join(shotDir, name), Buffer.from(r.result.data, 'base64'));
    };

    // ---------- 场景 1：默认模式 + 浏览器内自测 ----------
    console.log('\n=== 场景 1：默认预算，页面初始化 + 浏览器内自测 ===');
    await navigate(`${APP_BASE}/?selftest=1`);
    const ready = await waitForLine('SELFTEST_DONE');
    const doneLine = consoleLines.find((l) => l.includes('SELFTEST_DONE')) ?? '';
    assert('自测在浏览器中执行完成', ready, doneLine || '超时未收到 SELFTEST_DONE');
    const m = doneLine.match(/pass=(\d+) fail=(\d+)/);
    assert('浏览器自测 0 失败', !!m && m[2] === '0', doneLine);
    if (m) console.log(`    浏览器自测通过 ${m[1]} 项`);
    assert(
      '无未捕获异常',
      !consoleLines.some((l) => l.startsWith('UNCAUGHT')),
      consoleLines.find((l) => l.startsWith('UNCAUGHT')) ?? '',
    );

    await sleep(1500); // 等若干渲染帧
    const glInfo = await evaluate(`(() => {
      const r = window.__skinning.getRenderer();
      const c = document.createElement('canvas');
      const gl = c.getContext('webgl2');
      return JSON.stringify({
        webgl2: !!gl,
        gpuMaxVec: gl ? gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS) : 0,
        budget: r.getBudgetInfo(),
        drawCalls: r.drawCallCount,
      });
    })()`);
    const info = JSON.parse(glInfo);
    assert('WebGL2 上下文可用', info.webgl2 === true);
    assert('GPU MAX_VERTEX_UNIFORM_VECTORS 已读取', info.gpuMaxVec >= 16, `实测 ${info.gpuMaxVec}`);
    assert('默认模式单批渲染（draw call=1）', info.drawCalls === 1, `draw call=${info.drawCalls}`);
    assert('状态机已挂载', (await evaluate('!!window.__skinning.fsm')) === true);
    await screenshot('01-default-render.png');

    // ---------- 场景 2：强制低预算，分批降级 ----------
    console.log('\n=== 场景 2：?forcelimit=1 强制 32 vectors，分批降级 ===');
    await navigate(`${APP_BASE}/?forcelimit=1`);
    await waitForLine('draw call=', 10000).catch(() => {});
    await sleep(1500);
    const lowInfo = JSON.parse(
      await evaluate(`(() => {
        const r = window.__skinning.getRenderer();
        return JSON.stringify({ budget: r.getBudgetInfo(), drawCalls: r.drawCallCount, warnings: r.warnings });
      })()`),
    );
    assert('强制预算生效（32 vectors）', lowInfo.budget.maxVertexUniformVectors === 32);
    assert('单批上限降为 3 根骨骼', lowInfo.budget.maxBonesPerBatch === 3, `实际 ${lowInfo.budget.maxBonesPerBatch}`);
    assert('needsBatching 为 true', lowInfo.budget.needsBatching === true);
    assert('draw call 增加（分批渲染生效）', lowInfo.drawCalls > 1, `draw call=${lowInfo.drawCalls}`);
    assert('产生了可读警告', lowInfo.warnings.length >= 1, lowInfo.warnings.join('；'));
    const hasWarnLog =
      logEntries.some((t) => t.includes('分批')) || consoleLines.some((l) => l.includes('分批'));
    assert('console 输出降级警告', hasWarnLog);
    await screenshot('02-batched-render.png');

    // ---------- 场景 3：切换 walk，观察 crossfade 权重 ----------
    console.log('\n=== 场景 3：idle -> walk Crossfade 权重演化 ===');
    await navigate(APP_BASE);
    await sleep(800);
    await evaluate('window.__skinning.fsm.play("walk", 0.6)');
    // 轮询直到权重进入 0.5 附近（fade 时长 0.6s，中点权重 0.5/0.5）
    let midSnap = null;
    for (let i = 0; i < 40; i++) {
      midSnap = JSON.parse(await evaluate('JSON.stringify(window.__skinning.fsm.snapshot())'));
      const wi2 = midSnap.sources.find((s) => s.name === 'idle')?.weight ?? -1;
      const ww2 = midSnap.sources.find((s) => s.name === 'walk')?.weight ?? -1;
      if (Math.abs(wi2 - 0.5) < 0.05 && Math.abs(ww2 - 0.5) < 0.05) break;
      await sleep(30);
    }
    const wi = midSnap.sources.find((s) => s.name === 'idle')?.weight ?? -1;
    const ww = midSnap.sources.find((s) => s.name === 'walk')?.weight ?? -1;
    assert('中点两个片段同时活跃', midSnap.sources.length === 2, `源数=${midSnap.sources.length}`);
    assert('中点权重接近 0.5/0.5', Math.abs(wi - 0.5) < 0.05 && Math.abs(ww - 0.5) < 0.05, `idle=${wi.toFixed(3)} walk=${ww.toFixed(3)}`);
    await sleep(700); // fade 结束
    const endSnap = JSON.parse(await evaluate('JSON.stringify(window.__skinning.fsm.snapshot())'));
    assert('fade 结束后仅剩 walk', endSnap.sources.length === 1 && endSnap.primary === 'walk');
    await screenshot('03-walk-pose.png');
    // 走到 walk 极值时刻（关键帧 t=0.5s）再截一张，确认大姿态变形
    await evaluate('new Promise((resolve) => { const t0 = performance.now(); const tick = () => { const s = window.__skinning.fsm.snapshot(); const w = s.sources[0]; if (w && w.name === "walk" && Math.abs(w.time - 0.5) < 0.03) resolve(); else if (performance.now() - t0 < 3000) requestAnimationFrame(tick); else resolve(); }; tick(); })');
    const extreme = JSON.parse(await evaluate('JSON.stringify(window.__skinning.fsm.snapshot())'));
    assert('walk 极值帧可被采样到', extreme.sources.length === 1 && extreme.sources[0].name === 'walk');
    await screenshot('04-walk-extreme.png');

    ws.close();
    console.log(`\nCDP 验证结束：${failures === 0 ? '全部通过' : `失败 ${failures} 项`}`);
  } finally {
    chrome.kill();
  }
  process.exitCode = failures === 0 ? 0 : 1;
}

main().catch((e) => {
  console.error('CDP 验证脚本异常：', e);
  process.exitCode = 1;
});
