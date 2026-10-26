// @ts-check

// remark-math 6 parses `$$…$$` written on a single line as *text* math
// (`inlineMath`), not flow math — so a line like `$$\sum_i x_i$$` renders in
// textstyle, left-aligned, with no `.katex-display` wrapper (and no overflow
// scroll box on mobile). Only the fenced form
//
//   $$
//   \sum_i x_i
//   $$
//
// becomes a `math` node. This plugin promotes every `$$`-delimited
// `inlineMath` node that sits on a line of its own to a display `math` node,
// splitting the surrounding paragraph when prose precedes or follows it on
// adjacent lines. `$…$`, and `$$…$$` embedded mid-sentence, stay inline.
//
// Must run after remark-math (it consumes remark-math's nodes) and before
// rehype-katex (which keys display mode off the `math-display` class).

/** @typedef {import('mdast').Root} Root */
/** @typedef {import('mdast').Parent} Parent */
/** @typedef {import('mdast').Paragraph} Paragraph */
/** @typedef {import('mdast').RootContent} RootContent */
/** @typedef {import('mdast').PhrasingContent} PhrasingContent */
/** @typedef {import('mdast').InlineMath} InlineMath */
/** @typedef {import('mdast').Math} Math */

const LINE_END = /\n[ \t]*$/;
const LINE_START = /^[ \t]*\n/;

/** @returns {(tree: Root, file: import('vfile').VFile) => void} */
export default function remarkDisplayMath() {
  return (tree, file) => {
    const source = String(file.value);

    /** @param {PhrasingContent | undefined} node */
    const isDoubleDollar = (node) =>
      node?.type === 'inlineMath' &&
      node.position?.start.offset !== undefined &&
      source.startsWith('$$', node.position.start.offset);

    /**
     * Split a paragraph around its own-line `$$` math, or return null if it
     * has none.
     * @param {Paragraph} paragraph
     * @returns {RootContent[] | null}
     */
    const split = (paragraph) => {
      const kids = paragraph.children;
      // Decide every node against the original text before trimming any, so
      // back-to-back `$$…$$` lines each still see their newline neighbors.
      const ownLine = kids.map((node, i) => {
        const prev = kids[i - 1];
        const next = kids[i + 1];
        return (
          isDoubleDollar(node) &&
          (prev === undefined || (prev.type === 'text' && LINE_END.test(prev.value))) &&
          (next === undefined || (next.type === 'text' && LINE_START.test(next.value)))
        );
      });
      if (!ownLine.includes(true)) return null;

      /** @type {RootContent[]} */
      const out = [];
      /** @type {PhrasingContent[]} */
      let run = [];
      const flush = () => {
        if (run.length) out.push({ type: 'paragraph', children: run });
        run = [];
      };

      kids.forEach((node, i) => {
        if (ownLine[i]) {
          flush();
          out.push(toDisplayMath(/** @type {InlineMath} */ (node)));
          return;
        }
        if (node.type === 'text') {
          let value = node.value;
          if (ownLine[i - 1]) value = value.replace(LINE_START, '');
          if (ownLine[i + 1]) value = value.replace(LINE_END, '');
          if (value !== '') run.push({ ...node, value });
          return;
        }
        run.push(node);
      });
      flush();
      return out;
    };

    /** @param {Parent} parent */
    const walk = (parent) => {
      for (let i = 0; i < parent.children.length; i++) {
        const child = parent.children[i];
        if (child.type === 'paragraph') {
          const replacement = split(child);
          if (replacement) {
            parent.children.splice(i, 1, .../** @type {any[]} */ (replacement));
            i += replacement.length - 1;
          }
          continue;
        }
        if ('children' in child) walk(child);
      }
    };

    walk(tree);
  };
}

/**
 * Build the same `math` node shape mdast-util-math emits for a fenced block.
 * @param {InlineMath} node
 * @returns {Math}
 */
function toDisplayMath(node) {
  return {
    type: 'math',
    meta: null,
    value: node.value,
    position: node.position,
    data: {
      hName: 'pre',
      hChildren: [
        {
          type: 'element',
          tagName: 'code',
          properties: { className: ['language-math', 'math-display'] },
          children: [{ type: 'text', value: node.value }],
        },
      ],
    },
  };
}
