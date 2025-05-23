module.exports = {
  apps: [
    {
      name: "voice-agent-gpu",
      script: "./start_app_vc_gpu.sh",
      cwd: "/root/VoiceAgent",
      interpreter: "/bin/bash",
      env: {
        CONDA_DEFAULT_ENV: "va",
        OPENAI_API_KEY: "sk-xxxx",
      },
    },
    // 可以继续添加更多的服务配置
  ],
};
