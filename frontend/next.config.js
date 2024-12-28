/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // 设置输出为静态导出
  experimental: {
    appDir: true, // 启用 App Router
  },
};

module.exports = nextConfig;
