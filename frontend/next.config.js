const CopyPlugin = require("copy-webpack-plugin");

const isProd = process.env.NODE_ENV === 'production';

const internalHost = process.env.TAURI_DEV_HOST || 'localhost';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Transpile the specific packages you need
  transpilePackages: ['onnxruntime-web', '@ricky0123/vad-web'],

  // 注意：在 SSG 模式下使用 Next.js 的 Image 组件需要此功能。
  // 请参阅 https://nextjs.org/docs/messages/export-image-api 了解不同的解决方法。
  images: {
      unoptimized: true,
  },
  // 配置 assetPrefix，否则服务器无法正确解析您的资产。
  assetPrefix: isProd ? undefined : `http://${internalHost}:3000`,

  // Webpack custom configuration
  webpack: (config) => {
    // Add support for TypeScript files (.ts, .tsx)
    config.resolve.extensions.push(".ts", ".tsx");

    // Fallback for Node.js modules (like `fs`) that Next.js can't handle natively in the browser
    config.resolve.fallback = { fs: false };

    // Add CopyPlugin to handle static file copying
    config.plugins.push(
      new CopyPlugin({
        patterns: [
          {
            from: "node_modules/onnxruntime-web/dist/*.wasm",
            to: "../public/[name][ext]",
          },
          {
            from: "node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js",
            to: "../public/[name][ext]",
          },
          {
            from: "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
            to: "../public/[name][ext]",
          },
          {
            from: "node_modules/@ricky0123/vad-web/dist/*.onnx",
            to: "../public/[name][ext]",
          },
        ],
      })
    );

    return config;
  },
};

module.exports = nextConfig;
