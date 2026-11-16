import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { getAllPosts, getTopics, topicSlug } from "@/lib/posts";
import { PostList } from "@/components";

type Props = { params: Promise<{ topic: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
  return getTopics().map((t) => ({ topic: t.slug }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { topic } = await params;
  const t = getTopics().find((x) => x.slug === topic);
  return t ? { title: t.name } : {};
}

export default async function TopicPage({ params }: Props) {
  const { topic } = await params;
  const t = getTopics().find((x) => x.slug === topic);
  if (!t) notFound();
  const posts = getAllPosts().filter((p) => topicSlug(p.topic) === topic);
  return (
    <>
      <p className="back">
        <Link href="/topics/">← All topics</Link>
      </p>
      <h1 className="page-title">{t.name}</h1>
      <PostList posts={posts} />
    </>
  );
}
