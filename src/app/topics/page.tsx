import type { Metadata } from "next";
import Link from "next/link";
import { getTopics } from "@/lib/posts";

export const metadata: Metadata = { title: "Topics" };

export default function Topics() {
  return (
    <>
      <h1 className="page-title">Topics</h1>
      <ul className="topic-list">
        {getTopics().map((t) => (
          <li key={t.slug}>
            <Link href={`/topics/${t.slug}/`}>{t.name}</Link>
            <span>
              {t.count} {t.count === 1 ? "article" : "articles"}
            </span>
          </li>
        ))}
      </ul>
    </>
  );
}
