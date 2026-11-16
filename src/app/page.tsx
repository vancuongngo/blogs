import { getAllPosts, getTopics, topicSlug } from "@/lib/posts";
import { PostList } from "@/components";
import { site } from "@/lib/site";
import Link from "next/link";

export default function Home() {
  const posts = getAllPosts();
  const topics = getTopics();
  return (
    <>
      <section className="hero">
        <h1>{site.description}</h1>
        <div className="chips">
          {topics.map((t) => (
            <Link key={t.slug} href={`/topics/${topicSlug(t.name)}/`} className="chip">
              {t.name} <small>{t.count}</small>
            </Link>
          ))}
        </div>
      </section>
      <PostList posts={posts} />
    </>
  );
}
