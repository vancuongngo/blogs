import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeSlug from "rehype-slug";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeStringify from "rehype-stringify";
import rehypeShiki from "@shikijs/rehype";

const POSTS_DIR = path.join(process.cwd(), "content", "posts");

export type PostMeta = {
  slug: string;
  title: string;
  date: string; // ISO yyyy-mm-dd
  topic: string;
  tags: string[];
  summary: string;
  readingMinutes: number;
};

export type Post = PostMeta & { html: string };

export const topicSlug = (topic: string) =>
  topic.toLowerCase().trim().replace(/\s+/g, "-");

function parse(file: string) {
  const slug = file.replace(/\.md$/, "");
  const raw = fs.readFileSync(path.join(POSTS_DIR, file), "utf8");
  const { data, content } = matter(raw);

  for (const key of ["title", "date", "topic", "summary"]) {
    if (!data[key]) throw new Error(`${file}: missing front matter "${key}"`);
  }

  // gray-matter parses bare YAML dates into Date objects; normalise to a string.
  const date =
    data.date instanceof Date
      ? data.date.toISOString().slice(0, 10)
      : String(data.date);

  const words = content.split(/\s+/).filter(Boolean).length;
  const meta: PostMeta = {
    slug,
    title: String(data.title),
    date,
    topic: String(data.topic),
    tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
    summary: String(data.summary),
    readingMinutes: Math.max(1, Math.round(words / 220)),
  };
  return { meta, content };
}

function files() {
  return fs.readdirSync(POSTS_DIR).filter((f) => f.endsWith(".md"));
}

export function getAllPosts(): PostMeta[] {
  return files()
    .map((f) => parse(f).meta)
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

export async function getPost(slug: string): Promise<Post> {
  const { meta, content } = parse(`${slug}.md`);
  const html = String(
    await unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkRehype)
      .use(rehypeShiki, {
        themes: { light: "github-light", dark: "github-dark" },
        defaultColor: false, // emit --shiki-light/--shiki-dark vars; CSS picks via prefers-color-scheme
        fallbackLanguage: "text",
      })
      .use(rehypeSlug)
      .use(rehypeAutolinkHeadings, {
        behavior: "append",
        properties: { className: ["anchor"], ariaLabel: "Link to section" },
        content: { type: "text", value: "#" },
      })
      .use(rehypeStringify)
      .process(content),
  );
  return { ...meta, html };
}

export function getTopics() {
  const map = new Map<string, { name: string; slug: string; count: number }>();
  for (const p of getAllPosts()) {
    const slug = topicSlug(p.topic);
    const t = map.get(slug) ?? { name: p.topic, slug, count: 0 };
    t.count += 1;
    map.set(slug, t);
  }
  return [...map.values()].sort((a, b) => a.name.localeCompare(b.name));
}

export function formatDate(iso: string) {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
}
