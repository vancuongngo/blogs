// Project site: https://vancuongngo.github.io/blogs
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "/blogs";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
