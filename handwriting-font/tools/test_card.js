// Smoke test: open the card tool, wait for render, screenshot, and exercise PNG export.
const { chromium } = require('playwright');
(async () => {
  const [html, shot] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('file://' + html);
  await page.waitForSelector('#card svg text');
  const n = await page.$$eval('#card svg text', els => els.length);
  const status = await page.textContent('#status');
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#png')]);
  const p = await dl.path();
  const size = require('fs').statSync(p).size;
  await page.screenshot({ path: shot });
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  console.log(JSON.stringify({ glyphs: n, status, pngBytes: size, mobileHorizontalOverflow: overflow, errors }));
  await browser.close();
})();
