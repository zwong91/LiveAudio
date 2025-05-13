module.exports = {
  apps: [
    {
      name: "voice-agent-edge",
      script: "./start_app_edge.sh",
      cwd: "/root/voice-agent",
      interpreter: "/bin/bash",
      env: {
        CONDA_DEFAULT_ENV: "rt",
        OPENAI_API_KEY: "sk-xxxx" // 替换为你的 OpenAI API Key
      },
    },
    {
      name: "voice-agent-assets-gpu",
      script: "./start_app_vc_gpu.sh",
      cwd: "/root/rt-audio",
      interpreter: "/bin/bash",
      env: {
        CONDA_DEFAULT_ENV: "rt",
        OPENAI_API_KEY: "sk-xxxx"
      },
    },
    // 可以继续添加更多的服务配置
  ],
};
