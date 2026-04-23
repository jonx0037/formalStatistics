// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExternalLinks from 'rehype-external-links';

// https://astro.build/config
export default defineConfig({
  site: 'https://formalstatistics.com',
  integrations: [react(), mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypeExternalLinks, { target: '_blank', rel: ['noopener', 'noreferrer'] }],
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
