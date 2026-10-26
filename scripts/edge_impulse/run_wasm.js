// Run an Edge Impulse WebAssembly export (node/ folder of the deployment zip) over WAV
// clips, hop by hop as `dad wav` does: 1 s window, 0.125 s hop, decision stamped at the
// window's end. Writes file,t_end_s,score (score = the "drone" class value).
//
//   node scripts/edge_impulse/run_wasm.js <wasm node dir> <folder of 16 kHz WAVs> <out.csv>
const fs = require('fs');
const path = require('path');

const [wasmDir, folder, outCsv] = process.argv.slice(2);
if (!outCsv) { console.error('usage: run_wasm.js <wasm node dir> <wav folder> <out.csv>'); process.exit(1); }
const Module = require(path.resolve(wasmDir, 'edge-impulse-standalone.js'));

const SR = 16000, WIN = 16000, HOP = 2000;

function readWav(file) {
  // 16-bit PCM mono, as export_edge_impulse.py writes it; walks the chunks to find "data".
  const b = fs.readFileSync(file);
  if (b.toString('ascii', 0, 4) !== 'RIFF' || b.readUInt16LE(22) !== 1 || b.readUInt32LE(24) !== SR || b.readUInt16LE(34) !== 16)
    throw new Error(`${file}: need 16 kHz mono 16-bit PCM`);
  let p = 12;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), n = b.readUInt32LE(p + 4);
    if (id === 'data') {
      const x = new Float32Array(n / 2);
      for (let i = 0; i < x.length; i++) x[i] = b.readInt16LE(p + 8 + 2 * i); // raw int16 values, as Studio feeds them
      return x;
    }
    p += 8 + n + (n % 2);
  }
  throw new Error(`${file}: no data chunk`);
}

function classify(win) {
  const bytes = win.length * 4, ptr = Module._malloc(bytes);
  new Float32Array(Module.HEAPU8.buffer, ptr, win.length).set(win);
  const ret = Module.run_classifier(ptr, win.length, false);
  Module._free(ptr);
  if (ret.result !== 0) throw new Error('run_classifier failed: ' + ret.result);
  let drone = null;
  for (let i = 0; i < ret.size(); i++) { const c = ret.get(i); if (c.label === 'drone') drone = c.value; c.delete(); }
  if (drone === null) throw new Error('no "drone" class in the model');
  return drone;
}

Module.onRuntimeInitialized = () => {
  const r = Module.init();
  if (typeof r === 'number' && r !== 0) throw new Error('init failed: ' + r);
  const files = fs.readdirSync(folder).filter(f => f.endsWith('.wav')).sort();
  const lines = ['file,t_end_s,score'];
  for (const f of files) {
    const x = readWav(path.join(folder, f));
    for (let s = 0; s + WIN <= x.length; s += HOP)
      lines.push(`${path.join(folder, f)},${((s + WIN) / SR).toFixed(4)},${classify(x.subarray(s, s + WIN))}`);
  }
  fs.writeFileSync(outCsv, lines.join('\n') + '\n');
  console.log(`${files.length} files, ${lines.length - 1} windows -> ${outCsv}`);
};
