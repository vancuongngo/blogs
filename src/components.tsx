import Link from "next/link";
import { formatDate, topicSlug, type PostMeta } from "@/lib/posts";

export function PostList({ posts }: { posts: PostMeta[] }) {
  return (
    <ul className="post-list">
      {posts.map((p) => (
        <li key={p.slug}>
          <article>
            <div className="meta">
              <Link href={`/topics/${topicSlug(p.topic)}/`} className="topic">
                {p.topic}
              </Link>
              <span>·</span>
              <time dateTime={p.date}>{formatDate(p.date)}</time>
              <span>·</span>
              <span>{p.readingMinutes} min read</span>
            </div>
            <h2>
              <Link href={`/posts/${p.slug}/`}>{p.title}</Link>
            </h2>
            <p>{p.summary}</p>
          </article>
        </li>
      ))}
    </ul>
  );
}
