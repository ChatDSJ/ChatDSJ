#!/usr/bin/env node
/**
 * populate.js  ▸  node populate_root.js tests_populate.md
 */
const fs   = require('fs');
const path = require('path');

const input = process.argv[2] || 'tests_populate.md';
if (!fs.existsSync(input)) {
  console.error('❌  Markdown file not found:', input);
  process.exit(1);
}

const md = fs.readFileSync(input, 'utf8');

//  ### `path/to/file` … ```(optional-lang)\n(code)…```
const re = /###\s+`([^`]+)`[^]*?```[\w-]*\n([\s\S]*?)```/g;

let m, created = 0;
while ((m = re.exec(md)) !== null) {
  const filePath = m[1].trim();
  const code     = m[2].replace(/\s*$/, '') + '\n';

  const abs = path.resolve(filePath);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, code);

  console.log('✅  wrote', filePath);
  created++;
}

console.log(`\nFinished. ${created} files populated from ${input}.`);
