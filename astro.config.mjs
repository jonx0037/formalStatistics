// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from 'rehype-external-links';
import remarkDisplayMath from './src/plugins/remark-display-math.mjs';

// MDX's JSX path turns a GFM table cell's `align` into `style="text-align:…"`, but `optimize`
// serializes static subtrees as plain HTML and would keep the obsolete `align` attribute.
// Converting it up front keeps the optimized output identical to the JSX output.
function rehypeTableCellAlignToStyle() {
  /** @param {any} node */
  const walk = (node) => {
    const props = node.properties;
    if ((node.tagName === 'td' || node.tagName === 'th') && props?.align) {
      props.style = `text-align:${props.align}` + (props.style ? `;${props.style}` : '');
      delete props.align;
    }
    node.children?.forEach(walk);
  };
  return walk;
}

// https://astro.build/config
export default defineConfig({
  site: 'https://formalstatistics.com',
  // optimize: collapse static MDX subtrees (KaTeX output above all) into HTML strings.
  // Without it every KaTeX span becomes a JSX call and the SSR bundle peaked at ~10.4 GB RSS;
  // with it the build peaks at ~2.7 GB and the rendered pages are unchanged.
  integrations: [react(), mdx({ optimize: true })],
  markdown: {
    // remarkDisplayMath promotes single-line `$$…$$` to display math; see the
    // plugin header for why remark-math alone leaves them inline.
    remarkPlugins: [remarkMath, remarkDisplayMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
      // Must stay here: an `mdx({ rehypePlugins })` list replaces this one instead of extending it.
      rehypeTableCellAlignToStyle,
    ],
  },
  // Serialize page rendering so only one KaTeX-heavy MDX is in flight at a
  // time during SSR. Topics with 100+ `$$` blocks (e.g. Topic 30 with 112)
  // allocate thousands of transient KaTeX DOM trees per page; rendering
  // multiple concurrently multiplies peak memory and can OOM on Vercel's
  // 8 GB standard tier. Trades a modest build-time increase for a much
  // lower memory ceiling. (Added after Topic 30 OOM on PR 34, 2026-04-23.)
  build: {
    concurrency: 1,
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
