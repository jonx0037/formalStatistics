#!/usr/bin/env node
// One-shot static audit of internal /topics/* links and #section-N-X anchors
// across MDX, Astro, and curriculum-graph.json. Exits 1 if orphans exist.
// Run: node scripts/check-internal-links.mjs
//
// Note: uses regex.exec() for pattern matching. No child_process / shell.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const TOPICS_DIR = join(ROOT, 'src/content/topics');
const PAGES_DIR = join(ROOT, 'src/pages');
const COMPONENTS_DIR = join(ROOT, 'src/components');
const GRAPH_JSON = join(ROOT, 'src/data/curriculum-graph.json');

const slugs = new Set(
  readdirSync(TOPICS_DIR)
    .filter((f) => f.endsWith('.mdx'))
    .map((f) => f.replace(/\.mdx$/, ''))
);

const anchorsBySlug = new Map();
for (const slug of slugs) {
  const content = readFileSync(join(TOPICS_DIR, `${slug}.mdx`), 'utf8');
  const anchors = new Set();
  const anchorRe = /<a\s+id="([^"]+)"\s*(?:>\s*<\/a>|\/>)/g;
  let match;
  while ((match = anchorRe.exec(content)) !== null) anchors.add(match[1]);
  anchorsBySlug.set(slug, anchors);
}

function walkFiles(dir, exts) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) out.push(...walkFiles(full, exts));
    else if (exts.some((e) => entry.endsWith(e))) out.push(full);
  }
  return out;
}

const orphans = [];

function checkUrl(url, source, lineNo) {
  const [pathPart, ...anchorParts] = url.split('#');
  const anchor = anchorParts.join('#') || null;
  const slug = pathPart.replace(/^\/topics\//, '').replace(/\/$/, '');
  if (!slug) return;
  if (!slugs.has(slug)) {
    orphans.push({
      source,
      line: lineNo,
      url,
      reason: `slug "${slug}" not found under src/content/topics/`,
    });
    return;
  }
  if (anchor) {
    const anchors = anchorsBySlug.get(slug);
    if (!anchors.has(anchor)) {
      orphans.push({
        source,
        line: lineNo,
        url,
        reason: `anchor "#${anchor}" not found in ${slug}.mdx`,
      });
    }
  }
}

function scan(filePath) {
  const rel = relative(ROOT, filePath);
  const content = readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const mdLinkRe = /\]\((\/topics\/[^)\s]+)\)/g;
  const attrRe = /(?:href|url)="(\/topics\/[^"\s]+)"/g;
  lines.forEach((line, i) => {
    let match;
    while ((match = mdLinkRe.exec(line)) !== null) checkUrl(match[1], rel, i + 1);
    while ((match = attrRe.exec(line)) !== null) checkUrl(match[1], rel, i + 1);
  });
}

const targets = [
  ...walkFiles(TOPICS_DIR, ['.mdx']),
  ...walkFiles(PAGES_DIR, ['.astro']),
  ...walkFiles(COMPONENTS_DIR, ['.astro', '.tsx', '.ts']),
];
for (const f of targets) scan(f);

const graph = JSON.parse(readFileSync(GRAPH_JSON, 'utf8'));
for (const node of graph.nodes) {
  if (typeof node.url !== 'string' || !node.url.startsWith('/topics/')) continue;
  const slug = node.url.replace(/^\/topics\//, '').replace(/\/$/, '');
  if (!slugs.has(slug)) {
    orphans.push({
      source: 'src/data/curriculum-graph.json',
      line: 0,
      url: node.url,
      reason: `node "${node.id}" url slug "${slug}" not found under src/content/topics/`,
    });
  }
}

if (orphans.length === 0) {
  console.log(`PASS  checked ${targets.length} source files + ${graph.nodes.length} graph nodes across ${slugs.size} topic slugs — no orphan internal links`);
  process.exit(0);
}

console.log(`FAIL  ${orphans.length} orphan internal link(s):\n`);
for (const o of orphans) {
  console.log(`  ${o.source}:${o.line}`);
  console.log(`    ${o.url}`);
  console.log(`    ${o.reason}\n`);
}
process.exit(1);
