# VoiceAgent

Welcome to the VoiceAgent repository! This project hosts exciting applications leveraging advanced audio understanding and speech generation models to bring your audio experiences to life.

## Install

### Prerequisites

```sh
# System dependencies (Ubuntu/Debian)
apt update
apt install libsox-dev espeak-ng ffmpeg libopenblas-dev vim git-lfs \
    build-essential cmake libasound-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 -y

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create directory
mkdir -p /asset
chmod 777 /asset/

# Clone repository
git clone https://github.com/zwong91/VoiceAgent.git
cd VoiceAgent
```

### Environment Setup

```sh
# Install and setup miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash && source ~/miniconda3/bin/activate
conda config --set auto_activate_base false
conda create -n rt python=3.10 -y
conda activate rt

# Install dependencies using uv
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --force-reinstall
uv pip install -r requirements.txt

# Install XTTS
cd src/xtts
uv pip install -e ".[all,server,notebooks,bn,ja,ko,zh,languages]"

# Download XTTS-v2 model
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download coqui/XTTS-v2 --local-dir XTTS-v2

# pip install flash-attn

(rt) root@ash:~/audio# nvidia-smi
(rt) root@ash:~/audio# nvcc --version
```

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
    sudo docker run --gpus all -p 19999:19999 -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' VoiceAgent
    ```

## Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](generate_ssl.sh) microphone need ssl/tls

``` sh
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main --port 20000 --certfile cert.pem --keyfile private.key --tts-type xtts-v2 --vad-type pyannote --vad-args '{"auth_token": "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"}' --llm-type ollama
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

Resources
---------

* [WebRTC docs](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) - on <https://developer.mozilla.org>
- [Ollama](https://ollama.com/) - A local LLM inference engine for running Llama 3, Mistral, Gemma, and other LLMs
- [aiortc](https://aiortc.readthedocs.io/en/latest/) - A Python Library for WebRTC and ORTC communication
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).
