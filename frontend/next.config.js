const CopyPlugin = require("copy-webpack-plugin");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Transpile the specific packages you need
  transpilePackages: ['onnxruntime-web', '@ricky0123/vad-web'],

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
