import Link from "next/link";

export default function NotFound() {
  return (
    <>
      <h1 className="page-title">Page not found</h1>
      <p>
        <Link href="/">Back to all articles</Link>
      </p>
    </>
  );
}
