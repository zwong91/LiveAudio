const CopyPlugin = require("copy-webpack-plugin");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // 合并的 output 配置和 experimental 配置
  output: 'export', // 设置输出为静态导出
  experimental: {
    appDir: true, // 启用 App Router
  },

  // webpack 自定义配置
  webpack: (config, {}) => {
    config.resolve.extensions.push(".ts", ".tsx");
    config.resolve.fallback = { fs: false };

    // 添加 CopyPlugin 配置
    config.plugins.push(
      new CopyPlugin({
        patterns: [
          {
            from: "node_modules/onnxruntime-web/dist/*.wasm",
            to: "public/[name][ext]",
          },
          {
            from: "node_modules/@ricky0123/vad-web/dist/vad.worklet.bundle.min.js",
            to: "public/[name][ext]",
          },
          {
            from: "node_modules/@ricky0123/vad-web/dist/*.onnx",
            to: "public/[name][ext]",
          },
        ],
      })
    );

    return config;
  },
};

module.exports = nextConfig;
