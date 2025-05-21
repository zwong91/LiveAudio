<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Python Outbound Call Agent

<p>
  <a href="https://docs.livekit.io/agents/overview/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

This example demonstrates an full workflow of an AI agent that makes outbound calls. It uses LiveKit SIP and Python [Agents Framework](https://github.com/livekit/agents).

It can use a pipeline of STT, LLM, and TTS models, or a realtime speech-to-speech model. (such as ones from OpenAI and Gemini).

This example builds on concepts from the [Outbound Calls](https://docs.livekit.io/agents/start/telephony/#outbound-calls) section of the docs. Ensure that a SIP outbound trunk is configured before proceeding.

## Features

This example demonstrates the following features:

- Making outbound calls
- Detecting voicemail
- Looking up availability via function calling
- Transferring to a human operator
- Detecting intent to end the call
- Uses Krisp background voice cancellation to handle noisy environments

```
RunPod Pytorch 2.4.0
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate Python 3.12 virtual environment named 'va'
uv venv --python=python3.12 agent
source agent/bin/activate
which python
python --version

git clone https://github.com/zwong91/VoiceAgent.git
cd VoiceAgent
git checkout lk

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies using uv
uv pip install -r requirements.txt
python src/inbound_agent.py download-files

# Install ollama https://github.com/ollama/ollama/releases
apt update
apt install lshw jq vim curl ffmpeg libportaudio2 -y
apt-get -qq -y install espeak-ng > /dev/null 2>&1

(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &
ollama run gemma3:12b --verbose

```
***OpenAI plugin for LiveKit Agents***
平滑自建代码。。
SST  OpenAI 兼容 API  如 grok 的端点。 自建

TTS  也要兼容 OpenAI 端点。如 Kokoro Fast

LLM  Ollama 端点


Twilio SIP
SipTrunkID  or  BYOC Trunking ID
```bash
lk cloud auth
lk sip dispatch list
lk sip outbound list
lk sip inbound list

lk sip inbound create inbound-trunk.json
Using default project [voice-agent]
SIPTrunkID: ST_KQvebhnKoTRT

lk sip dispatch create dispatch-rule.json
Using default project [voice-agent]
SIPDispatchRuleID: SDR_zWq6fbvk43Pv

lk sip outbound create outbound-trunk.json
Using default project [voice-agent]
SIPTrunkID: ST_Vo4wWuadYrjx
```


https://github.com/remsky/Kokoro-FastAPI
```bash
docker run -itd -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest # CPU, or:
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest  #NVIDIA GPU

# Scratch
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI
uv venv
source .venv/bin/activate
uv sync --all-extras
# Models will auto-download, but if needed you can manually download:
# python docker/scripts/download_model.py --output api/src/models/v1_0
# Or run directly via UV:
./start-gpu.sh  # For GPU support
./start-cpu.sh  # For CPU support

```

***https://github.com/astramind-ai/Auralis**
```bash
uv venv --python=python3.10 xtts
source xtts/bin/activate

uv pip install auralis

auralis.openai --host 127.0.0.1 --port 8880 --model AstraMindAI/xttsv2 --gpt_model AstraMindAI/xtts2-gpt --max_concurrency 8 --vllm_logging_level warn

```

https://speaches.ai/usage/text-to-speech/
```bash
#CPU
curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml
curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cpu.yaml
export COMPOSE_FILE=compose.cpu.yaml
docker compose up -d

# CUDA
curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.yaml
curl --silent --remote-name https://raw.githubusercontent.com/speaches-ai/speaches/master/compose.cuda.yaml
export COMPOSE_FILE=compose.cuda.yaml

# Soure Code
git clone https://github.com/speaches-ai/speaches.git
cd speaches
uv venv
source .venv/bin/activate
uv sync --all-extras
uvicorn --factory --host 0.0.0.0 speaches.main:create_app

export SPEACHES_BASE_URL="http://localhost:8000"
# Listing all available STT models
uvx speaches-cli registry ls --task automatic-speech-recognition | jq '.data | [].id'
# Downloading a deepdml/faster-whisper-large-v3-turbo-ct2 model
uvx speaches-cli model download deepdml/faster-whisper-large-v3-turbo-ct2

```

要使用 turn-detector 、 silero 或 noise-cancellation 插件，首先需要下载模型文件

```bash
python src/inbound_agent.py download-files
# or
python src/outbound_agent.py download-files
```

## 运行
```bash
#1. console 模式下，代理在本地运行，仅在您的终端内可用。
#2. dev （开发/调试）或 start （生产）模式下运行您的代理，以连接到 LiveKit 并加入房间。
python src/inbound_agent.py  console/dev/start
```

Now, your worker is running, and waiting for dispatches in order to make outbound calls.

### Making a call

You can dispatch an agent to make a call by using the `lk` CLI:

```shell
# 创建一个新房间，并将您的代理分配到该房间，附带要拨打的电话号码。
lk dispatch create \
  --new-room \
  --agent-name outbound-caller \
  --metadata '{"phone_number": "+1234567890", "transfer_to": "+9876543210}'
```

```python
await lkapi.agent_dispatch.create_dispatch(
    api.CreateAgentDispatchRequest(
        # Use the agent name you set in the WorkerOptions
        agent_name="my-telephony-agent",

        # The room name to use. This should be unique for each call
        room=f"outbound-{''.join(str(random.randint(0, 9)) for _ in range(10))}",

        # Here we use JSON to pass the phone number, and could add more information if needed.
        metadata='{"phone_number": "+15105550123"}'
    )
)
```

### Q &A
2025-05-21 06:24:48,284:DEBUG:urllib3.connectionpool:_make_request:546:https://huggingface.co:443 "GET /api/models/deepdml/faster-whisper-large-v3-turbo-ct2/revision/main HTTP/11" 200 2407
2025-05-21 06:24:49,957:INFO:speaches.model_manager:_load:48:Model deepdml/faster-whisper-large-v3-turbo-ct2 loaded in 1.90s
2025-05-21 06:24:49,957:DEBUG:speaches.model_manager:_increment_ref:56:Incremented ref count for deepdml/faster-whisper-large-v3-turbo-ct2, self.ref_count=1
2025-05-21 06:24:49,957:INFO:faster_whisper:transcribe:839:Processing audio with duration 00:02.452
Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
Aborted (core dumped)


(speaches) root@780f16ccf37b:/workspace/speaches# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0

(speaches) root@780f16ccf37b:/workspace/speaches# nvidia-smi
Wed May 21 06:26:26 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 4000 Ada Gene...    On  |   00000000:C2:00.0 Off |                  Off |
| 30%   31C    P8             12W /  130W |   10607MiB /  20475MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+


# 安装 cuBLAS 和 cuDNN
apt install -y cuda-toolkit-12-4

***https://developer.nvidia.com/cudnn-downloads***
