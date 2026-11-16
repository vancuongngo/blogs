import type { Metadata } from "next";
import Link from "next/link";
import { site } from "@/lib/site";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: site.title, template: `%s · ${site.title}` },
  description: site.description,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="site-header">
          <div className="container header-inner">
            <Link href="/" className="brand">
              {site.title}
            </Link>
            <nav>
              <Link href="/">Articles</Link>
              <Link href="/topics/">Topics</Link>
              <a href={site.repo}>GitHub</a>
            </nav>
          </div>
        </header>
        <main className="container">{children}</main>
        <footer className="site-footer">
          <div className="container">Written to be read, shared, and corrected.</div>
        </footer>
      </body>
    </html>
  );
}
