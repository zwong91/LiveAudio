
LK Solutions:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create and activate Python 3.12 virtual environment named 'va'
uv venv --python=python3.12 agent
source agent/bin/activate
which python

uv sync

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies using uv
uv pip install -r requirements.txt

# Install ollama https://github.com/ollama/ollama/releases
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama run gemma3:12b --verbose

```
***OpenAI plugin for LiveKit Agents***
平滑自建代码。。
SST  OpenAI 兼容 API  如 grok 的端点。 自建

TTS  也要兼容 OpenAI 端点。如 Kokoro Fast

LLM  Ollama 端点


Twilio SIP
SipTrunkID  or  BYOC Trunking ID


https://github.com/remsky/Kokoro-FastAPI

https://speaches.ai/usage/text-to-speech/


要使用 turn-detector 、 silero 或 noise-cancellation 插件，首先需要下载模型文件

```bash
python src/main.py download-files
```

## 运行
```bash
#1. console 模式下，代理在本地运行，仅在您的终端内可用。
#2. dev （开发/调试）或 start （生产）模式下运行您的代理，以连接到 LiveKit 并加入房间。
python src/main.py  console/dev/start
```
