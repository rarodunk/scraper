// Render sample text in a font with Chromium (applies calt like real apps do).
// usage: node preview.js <font.otf> <out.png> "<line1>" ["<line2>" ...]
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [font, out, ...lines] = process.argv.slice(2);
  const b64 = fs.readFileSync(font).toString('base64');
  const html = `<html><head><style>
    @font-face { font-family: H; src: url(data:font/otf;base64,${b64}); }
    body { margin: 24px; background: #fffdf7; }
    p { font-family: H, monospace; font-size: 56px; margin: 6px 0; color: #1d2a5c; line-height: 1.35; }
    .grid { background: repeating-linear-gradient(#fffdf7 0 74px, #dde 74px 75px); }
  </style></head><body>${lines.map(l => `<p>${l}</p>`).join('')}</body></html>`;
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 200 }, deviceScaleFactor: 1 });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
})();
