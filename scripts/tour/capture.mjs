// Drive the app in Chrome and capture the tour's screenshots (raw, uncropped).
// Run by make_tour.py:  node capture.mjs <stage> <port> <outdir>
//   stage "home":  the Home page (before the demo label set exists)
//   stage "rest":  Label (with the demo set), Explorer, Compare, Survey (save + apply)
import { chromium } from 'playwright-core';
import fs from 'fs';
import path from 'path';

const [stage, port, out] = process.argv.slice(2);
const url = (p) => `http://localhost:${port}/${p}`;
const shot = (name) => path.join(out, name);
const b = await chromium.launch({ channel: 'chrome', headless: true });
const p = await b.newPage({ viewport: { width: 1500, height: 1600 } });
const settle = (ms = 8000) => p.waitForTimeout(ms);
// Where each named part of the page sat in each capture, for make_tour.py to
// place markers and crops by element rather than by fixed pixels.
const MEASURE = JSON.parse(fs.readFileSync(path.join(out, 'measure.json')));
async function save(name) {
  await p.screenshot({ path: shot(name) });
  const boxes = {};
  for (const sel of MEASURE[name] || []) {
    try { boxes[sel] = await p.locator(sel).first().boundingBox({ timeout: 2000 }); }
    catch { boxes[sel] = null; }
  }
  fs.writeFileSync(shot(name + '.boxes.json'), JSON.stringify(boxes, null, 1));
}
const log = (m) => console.log('captured:', m);
async function closeWelcome() {
  if (await p.locator('.st-key-w_close button').count()) {
    await p.locator('.st-key-w_close button').first().click(); await settle(3000);
  }
}

if (stage === 'home') {
  await p.goto(url('')); await p.getByText('Your datasets').first().waitFor({ timeout: 120000 });
  await settle(4000); await closeWelcome();
  await save('home.png'); log('home');
} else {
  const pts = JSON.parse(fs.readFileSync(path.join(out, 'pts.json')));
  // Label: draw a box over a call the experts labelled, and ask for suggestions.
  await p.setViewportSize({ width: 1500, height: 2300 });
  await p.goto(url('label')); await p.getByText('Label set you are working on').first().waitFor({ timeout: 120000 });
  await settle(5000);
  await p.getByRole('textbox', { name: 'Your name' }).fill('Mara'); await p.keyboard.press('Enter'); await settle(4000);
  if (!(await p.getByText(pts.set).first().isVisible())) {
    await p.getByRole('combobox', { name: /Label set you are working on/ }).click();
    await p.getByRole('option', { name: pts.set }).click(); await settle(9000);
  }
  const img = () => p.frameLocator('iframe[title="streamlit_image_coordinates.streamlit_image_coordinates"]').locator('img');
  async function clickAt(fx, fy) {
    const bb = await img().boundingBox(); await p.mouse.click(bb.x + bb.width * fx, bb.y + bb.height * fy); await settle(7000);
  }
  await clickAt(pts.x1, pts.y1); await clickAt(pts.x2, pts.y2);
  await p.getByRole('button', { name: /Suggest birds/ }).click(); await settle(9000);
  await save('label-chunk.png'); log('label');
  await p.getByRole('button', { name: 'Discard' }).click(); await settle(5000);
  await p.getByText('Plan and sample', { exact: true }).click(); await settle(9000);
  await save('label-plan.png'); log('plan');
  await p.getByText('Progress', { exact: true }).click(); await settle(10000);
  await save('label-progress.png'); log('progress');
  // Explorer
  await p.setViewportSize({ width: 1500, height: 1400 });
  await p.goto(url('explorer')); await p.getByText('Where the birdsong is').first().waitFor({ timeout: 180000 });
  await settle(8000); await save('explorer-navigator.png'); log('explorer');
  // Compare (one tall page)
  await p.setViewportSize({ width: 1500, height: 4200 });
  await p.goto(url('compare')); await p.getByText('Is one model better here?').first().waitFor({ timeout: 600000 });
  await settle(10000); await save('compare-full.png'); log('compare');
  // Survey: save a setup and apply it
  await p.setViewportSize({ width: 1500, height: 9000 });
  await p.goto(url('survey')); await p.getByText('Save this setup').first().waitFor({ timeout: 600000 });
  await settle(10000);
  await p.getByRole('textbox', { name: 'Name' }).fill(pts.setup); await p.keyboard.press('Enter'); await settle(4000);
  await p.getByRole('button', { name: 'Save setup' }).click(); await settle(10000);
  await p.getByRole('button', { name: 'Apply to every recording' }).click(); await settle(25000);
  await save('survey-full.png'); log('survey');
}
await b.close();
