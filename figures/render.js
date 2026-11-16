const { chromium } = require('playwright');
(async () => {
  const [html, out, W, H] = process.argv.slice(2);
  const b = await chromium.launch();
  for (const [sf, name] of [[1.5, out + '_preview.png'], [4, out + '.png']]) {
    const p = await b.newPage({ viewport: { width: +W, height: +H }, deviceScaleFactor: sf });
    await p.goto('file://' + html); await p.evaluate(() => document.fonts.ready);
    await p.screenshot({ path: name }); await p.close();
  }
  const p = await b.newPage({ viewport: { width: +W, height: +H } });
  await p.goto('file://' + html); await p.evaluate(() => document.fonts.ready);
  const mm = 190, s = mm / (+W * 25.4 / 96);
  await p.pdf({ path: out + '.pdf', width: mm + 'mm', height: (mm * H / W) + 'mm', scale: s, printBackground: true, pageRanges: '1' });
  await b.close();
})();
