import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { formatDate, getAllPosts, getPost, topicSlug } from "@/lib/posts";

type Props = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllPosts().map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params;
  const post = await getPost(slug).catch(() => null);
  if (!post) return {};
  return { title: post.title, description: post.summary };
}

export default async function PostPage({ params }: Props) {
  const { slug } = await params;
  const post = await getPost(slug).catch(() => null);
  if (!post) notFound();

  return (
    <article className="post">
      <header>
        <div className="meta">
          <Link href={`/topics/${topicSlug(post.topic)}/`} className="topic">
            {post.topic}
          </Link>
          <span>·</span>
          <time dateTime={post.date}>{formatDate(post.date)}</time>
          <span>·</span>
          <span>{post.readingMinutes} min read</span>
        </div>
        <h1>{post.title}</h1>
      </header>
      <div className="prose" dangerouslySetInnerHTML={{ __html: post.html }} />
      {post.tags.length > 0 && (
        <footer className="tags">
          {post.tags.map((t) => (
            <span key={t} className="chip static">
              {t}
            </span>
          ))}
        </footer>
      )}
      <p className="back">
        <Link href="/">← All articles</Link>
      </p>
    </article>
  );
}
