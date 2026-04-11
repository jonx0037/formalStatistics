import { defineCollection } from 'astro:content';
import { z } from 'astro:schema';
import { glob } from 'astro/loaders';

const topics = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: './src/content/topics' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    status: z.enum(['published', 'draft', 'planned']),
    difficulty: z.enum(['foundational', 'intermediate', 'advanced']),
    prerequisites: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    domain: z.string(),
    videoId: z.string().nullable().default(null),
    notebookPath: z.string().optional(),
    githubUrl: z.string().optional(),
    datePublished: z.coerce.date().optional(),
    estimatedReadTime: z.number().optional(),
    abstract: z.string().optional(),
    formalcalculusPrereqs: z
      .array(
        z.object({
          topic: z.string(),
          site: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    formalmlConnections: z
      .array(
        z.object({
          topic: z.string(),
          site: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    connections: z
      .array(
        z.object({
          topic: z.string(),
          relationship: z.string(),
        }),
      )
      .default([]),
    references: z
      .array(
        z.object({
          type: z.string(),
          title: z.string(),
          author: z.string(),
          year: z.number().optional(),
          edition: z.string().optional(),
          publisher: z.string().optional(),
          note: z.string().optional(),
        }),
      )
      .default([]),
  }),
});

export const collections = { topics };
