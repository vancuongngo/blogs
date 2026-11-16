# blogs

Articles on AI and other topics. Next.js (static export) deployed to GitHub Pages: https://vancuongngo.github.io/blogs/

## Writing a post

Add `content/posts/<slug>.md`:

```md
---
title: "Post title"
date: "2026-11-16"
topic: "AI"          # one topic per post; drives /topics/<topic>/
tags: ["LLM"]
summary: "One or two sentences shown in lists."
---

Markdown (GFM) body. Start sections at `##`; the title is the page `h1`.
```

## Develop

```bash
npm install
npm run dev      # http://localhost:3000/blogs
npm run build    # static site in ./out
```

## Deploy

Push to `main`. The workflow in `.github/workflows/deploy.yml` builds and publishes to Pages.
One-time: repo **Settings → Pages → Source: GitHub Actions**.

If the repo is ever renamed or moved to a user site (`<user>.github.io`), change `basePath` in `next.config.mjs`.
