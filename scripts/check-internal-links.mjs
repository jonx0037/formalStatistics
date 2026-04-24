#!/usr/bin/env node
// Static audit of internal topic links and #section-N-X anchors across MDX,
// Astro, component source, and data files. Exits 1 if orphans exist.
// Run: node scripts/check-internal-links.mjs
//
// What's checked:
//   - Cross-topic links:    [text](/topics/<slug>[#section-N-X])
//   - href/url attributes:  href="/topics/<slug>[#anchor]" or url="..."
//                           (double and single quotes)
//   - Same-page anchors:    [text](#section-N-X)  /  href="#section-N-X"
//   - curriculum-graph.json node.url slugs
//
// No shell. Only fs + regex. Uses matchAll() to avoid stateful lastIndex
// pitfalls on /g regexes.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const TOPICS_DIR = join(ROOT, 'src/content/topics');
const PAGES_DIR = join(ROOT, 'src/pages');
const COMPONENTS_DIR = join(ROOT, 'src/components');
const DATA_DIR = join(ROOT, 'src/data');
const GRAPH_JSON = join(ROOT, 'src/data/curriculum-graph.json');

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

// Accept any <a ... id="..."> opener — single or double quotes, id in any
// position, optional extra attributes, empty or self-closing form.
const ANCHOR_RE = /<a\s+[^>]*id=["']([^"']+)["'][^>]*>/g;

// Build slug set from all .mdx files under TOPICS_DIR (walks subdirs too, so
// if topics ever get organized into per-track folders, this still works).
const topicMdx = walkFiles(TOPICS_DIR, ['.mdx']);
const slugByMdxPath = new Map();
const slugs = new Set();
for (const filePath of topicMdx) {
  const rel = relative(TOPICS_DIR, filePath);
  const slug = rel.replace(/\\/g, '/').replace(/\.mdx$/, '');
  slugByMdxPath.set(filePath, slug);
  slugs.add(slug);
}

// For each topic, collect its anchor ids.
const anchorsBySlug = new Map();
for (const [filePath, slug] of slugByMdxPath) {
  const content = readFileSync(filePath, 'utf8');
  const anchors = new Set();
  for (const match of content.matchAll(ANCHOR_RE)) anchors.add(match[1]);
  anchorsBySlug.set(slug, anchors);
}

const orphans = [];

function pushOrphan(source, line, url, reason) {
  orphans.push({ source, line, url, reason });
}

function checkCrossTopic(url, source, lineNo) {
  const [pathPart, ...anchorParts] = url.split('#');
  const anchor = anchorParts.join('#') || null;
  const slug = pathPart.replace(/^\/topics\//, '').replace(/\/$/, '');
  if (!slug) return;
  if (!slugs.has(slug)) {
    pushOrphan(source, lineNo, url, `slug "${slug}" not found under src/content/topics/`);
    return;
  }
  if (anchor) {
    const anchors = anchorsBySlug.get(slug);
    if (!anchors.has(anchor)) {
      pushOrphan(source, lineNo, url, `anchor "#${anchor}" not found in ${slug}.mdx`);
    }
  }
}

function checkSamePage(anchor, source, lineNo, selfAnchors) {
  if (!selfAnchors) return;
  if (!selfAnchors.has(anchor)) {
    pushOrphan(source, lineNo, `#${anchor}`, `same-page anchor "#${anchor}" not defined in this file`);
  }
}

function scan(filePath) {
  const rel = relative(ROOT, filePath);
  const content = readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  // selfAnchors is only non-null for MDX topics (same-page refs only make
  // sense inside a single topic page; astro/tsx pages aren't topic bodies).
  const mdxSlug = slugByMdxPath.get(filePath);
  const selfAnchors = mdxSlug ? anchorsBySlug.get(mdxSlug) : null;

  const crossMdRe = /\]\((\/topics\/[^)\s]+)\)/g;
  const crossAttrRe = /(?:href|url)=["'](\/topics\/[^"'\s]+)["']/g;
  const samePageMdRe = /\]\((#section-[^)\s]+)\)/g;
  const samePageAttrRe = /(?:href|url)=["'](#section-[^"'\s]+)["']/g;

  lines.forEach((line, i) => {
    const lineNo = i + 1;
    for (const m of line.matchAll(crossMdRe)) checkCrossTopic(m[1], rel, lineNo);
    for (const m of line.matchAll(crossAttrRe)) checkCrossTopic(m[1], rel, lineNo);
    for (const m of line.matchAll(samePageMdRe)) {
      checkSamePage(m[1].slice(1), rel, lineNo, selfAnchors);
    }
    for (const m of line.matchAll(samePageAttrRe)) {
      checkSamePage(m[1].slice(1), rel, lineNo, selfAnchors);
    }
  });
}

const targets = [
  ...topicMdx,
  ...walkFiles(PAGES_DIR, ['.astro']),
  ...walkFiles(COMPONENTS_DIR, ['.astro', '.tsx', '.ts']),
  ...walkFiles(DATA_DIR, ['.ts', '.json']),
];
for (const f of targets) scan(f);

const graph = JSON.parse(readFileSync(GRAPH_JSON, 'utf8'));
for (const node of graph.nodes) {
  if (typeof node.url !== 'string' || !node.url.startsWith('/topics/')) continue;
  const slug = node.url.replace(/^\/topics\//, '').replace(/\/$/, '');
  if (!slugs.has(slug)) {
    pushOrphan(
      'src/data/curriculum-graph.json',
      0,
      node.url,
      `node "${node.id}" url slug "${slug}" not found under src/content/topics/`,
    );
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
