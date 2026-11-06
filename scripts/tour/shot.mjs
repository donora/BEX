// Screenshot one app page, to check a change by eye.
// node shot.mjs <port> <page> <out.png> [settle_ms] [click_x_frac]
// click_x_frac: also click the Explorer's figure this far across, then shoot again.
import { chromium } from 'playwright-core';

const [port, page, out, settle = '20000', clickX] = process.argv.slice(2);
const b = await chromium.launch({ channel: 'chrome', headless: true });
const p = await b.newPage({ viewport: { width: 1500, height: 2000 } });
await p.goto(`http://localhost:${port}/${page}`);
await p.waitForTimeout(Number(settle));
const close = p.locator('.st-key-w_close button');
if (await close.count()) { await close.first().click(); await p.waitForTimeout(6000); }
if (clickX) {
  const imgs = p.locator('[data-testid="stCustomComponentV1"]');
  const n = await imgs.count();
  console.log('components', n);
  for (let i = n - 1; i >= 0; i--) {
    const bb = await imgs.nth(i).boundingBox();
    if (bb && bb.height > 400) {
      await p.mouse.click(bb.x + bb.width * Number(clickX), bb.y + bb.height * 0.2);
      break;
    }
  }
  await p.waitForTimeout(Number(settle));
}
await p.screenshot({ path: out, fullPage: true });
await b.close();
