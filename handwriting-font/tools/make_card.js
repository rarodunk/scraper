// Render a finished card from a message file through the card tool (same layout, variation and wobble
// as the browser UI), and save a print-ready PDF plus a PNG preview.
//
//   node tools/make_card.js build/card.html message.txt build/my_card \
//        [--size 5x7|4x6|a2|a6|3.5x5] [--orient p|l] [--ink black|blue|ballpoint] \
//        [--fs 20] [--lh 1.45] [--wobble 100] [--align left|center] [--seed 7]
//
// Writes <out>.pdf (vector, page = card size, white) and <out>.png (300 dpi, white background).
// Exits non-zero if the text does not fit on the card.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const INK = { black: '#1a1a1a', blue: '#1d2a5c', ballpoint: '#1f3fa8' };

(async () => {
  const [html, msgFile, out, ...rest] = process.argv.slice(2);
  const opt = { size: '5x7', orient: 'l', ink: 'black', fs: '20', lh: '1.45', wobble: '100', align: 'left', seed: '7' };
  for (let i = 0; i < rest.length; i += 2) opt[rest[i].replace(/^--/, '')] = rest[i + 1];
  const message = fs.readFileSync(msgFile, 'utf8').replace(/\s+$/, '');

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('file://' + path.resolve(html));
  await page.waitForSelector('#card svg');

  // drive the same controls a person would use, then fix the variation seed for reproducibility
  const status = await page.evaluate(({ message, opt, ink }) => {
    const set = (id, v) => { const el = document.getElementById(id); el.value = v; };
    set('msg', message); set('size', opt.size); set('orient', opt.orient); set('ink', ink);
    set('fs', opt.fs); set('lh', String(Math.round(parseFloat(opt.lh) * 100))); set('mess', opt.wobble);
    set('align', opt.align);
    document.getElementById('transparent').checked = false;
    seed = parseInt(opt.seed, 10);  // top-level `let seed` in the card tool
    render();
    return document.getElementById('status').textContent;
  }, { message, opt, ink: INK[opt.ink] || opt.ink });
  if (errors.length) throw new Error(errors.join('; '));
  if (/runs off/.test(status)) {
    console.error(`Does not fit: ${status} Try a smaller --fs or --lh.`);
    process.exit(2);
  }
  if (status) console.log(`note: ${status}`);

  // PNG at 300 dpi through the tool's own exporter (white background); before print emulation,
  // which hides the controls
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#png')]);
  await dl.saveAs(`${out}.png`);

  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: `${out}.pdf`, preferCSSPageSize: true, printBackground: false });
  await browser.close();
  console.log(`wrote ${out}.pdf and ${out}.png`);
})();
