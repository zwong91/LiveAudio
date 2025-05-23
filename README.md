# VoiceAgent

Welcome to the VoiceAgent repository! This project hosts exciting applications leveraging advanced audio understanding and speech generation models to bring your audio experiences to life.

## Install

### Prerequisites

```sh

# Clone repository
git clone https://github.com/zwong91/VoiceAgent.git
cd VoiceAgent

# System dependencies (Ubuntu/Debian)
apt update
apt-get -qq -y install espeak-ng > /dev/null 2>&1
apt install curl lshw ffmpeg libopenblas-dev vim git-lfs \
    build-essential cmake libasound-dev portaudio19-dev \
    libportaudio2 -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

```

### Environment Setup

```sh
# Create and activate Python 3.10 virtual environment named 'va'
uv venv --python=python3.10 koko
source koko/bin/activate

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Install dependencies using uv
uv pip install -r requirements.txt

# Install ollama https://github.com/ollama/ollama/releases
# Install ollama https://github.com/ollama/ollama/releases
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &

ollama run gemma3:12b --verbose

# download files
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  livekit/turn-detector \
  --filename onnx/model_q8.onnx \
  --revision v0.2.0-intl

```

### Open an ngrok tunnel

When developing & testing locally, you'll need to open a tunnel to forward requests to your local development server. These instructions use ngrok.

Open a Terminal and run:

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok

ngrok config add-authtoken 2q0o5XSi73aG9m4KyMxHB0pmEXi_2mUYW4R1wDsanPzWgCWrW

ngrok http 8765
```

Once the tunnel has been opened, copy the `Forwarding` URL. It will look something like: `https://[your-ngrok-subdomain].ngrok.app`. You will need this when configuring your Twilio number setup.

Note that the `ngrok` command above forwards to a development server running on port `5050`, which is the default port configured in this application. If you override the `PORT` defined in `main.py`, you will need to update the `ngrok` command accordingly.

Keep in mind that each time you run the `ngrok http` command, a new URL will be created, and you'll need to update it everywhere it is referenced below.

## Docker Setup

1. Install NVIDIA Container Toolkit:

    To use GPU for model training and inference in Docker, you need to install NVIDIA Container Toolkit:

    For Ubuntu users:

    ```bash
    # Add repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # Install nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # Restart Docker service
    sudo systemctl restart docker
    ```

    For users of other Linux distributions, please refer to: [NVIDIA Container Toolkit Install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. You can build the container image with:

    ```shell
    sudo docker build -t VoiceAgent .
    ```

    After getting your VAD token (see next sections) run:

    ```bash
    sudo docker volume create huggingface

    sudo docker run --gpus all -p 8765:8765 -v huggingface:/root/.cache/huggingface  -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' VoiceAgent
    ```

    The "volume" stuff will allow you not to re-download the huggingface models each
    time you re-run the container. If you don't need this, just use:

    ```bash
    sudo docker run --gpus all -p 8765:8765 -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' VoiceAgent
    ```

## Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](generate_ssl.sh) microphone need ssl/tls

``` sh
# runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main --llm-type openai
```

***test***

```bash
export PYANNOTE_AUTH_TOKEN=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
ASR_TYPE=sensevoice python -m unittest test.server.test_server


```

## FAQ

1. "`GLIBCXX_3.4.32' not found" error at runtime. GCC 13.2.0***

[https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0]

2. How clone a voice submit the filename of a wave file containing the source voice

voice cloning works best with a 22050 Hz mono 16bit WAV file containing a short (~5-30 sec) sample of the target speaker's voice. The sample should be a clean recording with no background noise or music. The speaker should be speaking in a natural, conversational tone. The sample should be representative of the speaker's voice, including their accent, intonation, and speaking style.

3. Coqui AI XTTS-v2 tts 架构 high level

XTTS 利用 VQ-VAE 模型将音频离散化为音频标记。
它使用 GPT 模型根据输入文本和说话者潜变量speaker latents 预测这些音频标记。说话者潜变量speaker latents通过一系列自注意力层计算得出。
GPT 模型的输出被传递给解码器模型，输出音频信号。使用扩散模型将 GPT 输出转换为声谱图帧，然后利用 UnivNet 生成最终的音频信号。

4. Kokoro TTS Alternative to Coqui AI XTTS-v2, Kokoro TTS is a text-to-speech (TTS) system that uses a combination of neural networks and deep learning techniques to generate high-quality speech from text input. It is designed to produce natural-sounding speech with a focus on expressiveness and emotional intonation. Kokoro TTS is built on top of the Tacotron architecture, which is a sequence-to-sequence model that converts text to mel-spectrograms. The mel-spectrograms are then converted to audio waveforms using a vocoder, such as WaveGlow or HiFi-GAN.


Resources
---------

* [WebRTC docs](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) - on <https://developer.mozilla.org>

* [Ollama](https://ollama.com/) - A local LLM inference engine for running Llama 3, Mistral, Gemma, and other LLMs

* [aiortc](https://aiortc.readthedocs.io/en/latest/) - A Python Library for WebRTC and ORTC communication
* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).
