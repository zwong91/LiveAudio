# LiveAudio

Welcome to the LiveAudio repository! This project hosts A exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life, designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

## Install

**Clone and install**

- Clone the repo and submodules

``` sh

#0  source code

apt update
# (Ubuntu / Debian User) Install sox + ffmpeg
apt install libsox-dev ffmpeg libopenblas-dev vim git-lfs -y

# (Ubuntu / Debian User) Install pyaudio 
apt install build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python


mkdir /asset
chmod 777 /asset/
git clone https://github.com/zwong91/LiveAudio.git
cd /workspace/LiveAudio
git pull

#1 pre_install.sh
# 安装 miniconda, PyTorch/CUDA 的 conda 环境
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash && source ~/miniconda3/bin/activate
conda config --set auto_activate_base false
conda create -n rt python=3.10  -y
conda activate rt

#2  LiveAudio
cd /workspace/LiveAudio
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

#3 xtts
cd /workspace/LiveAudio/src/xtts
pip install -e .[all,dev,notebooks]  -i https://pypi.tuna.tsinghua.edu.cn/simple

#4. download xtts-v2 
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download coqui/XTTS-v2  --local-dir  XTTS-v2

(rt) root@ash:~/audio# nvidia-smi
(rt) root@ash:~/audio# nvcc --version
(rt) root@ash:~/audio# pip show torch
```

## Q
***"`GLIBCXX_3.4.32' not found" error at runtime. GCC 13.2.0***
```
https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0
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
    sudo docker build -t LiveAudio .
    ```

    After getting your VAD token (see next sections) run:

    ```bash
    sudo docker volume create huggingface

    sudo docker run --gpus all -p 8765:8765 -v huggingface:/root/.cache/huggingface  -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' LiveAudio
    ```

    The "volume" stuff will allow you not to re-download the huggingface models each
    time you re-run the container. If you don't need this, just use:

    ```bash
    sudo docker run --gpus all -p 19999:19999 -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' LiveAudio
    ```

## Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](generate_ssl.sh) microphone need ssl/tls

``` sh
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main --port 20000 --certfile cf.pem --keyfile cf.key --tts-type xtts-v2 --vad-type pyannote --vad-args '{"auth_token": "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"}' --llm-type ollama
```

***test***

```bash
export PYANNOTE_AUTH_TOKEN=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
ASR_TYPE=sensevoice python -m unittest test.server.test_server
```

Resources
---------
* [WebRTC docs](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) - on https://developer.mozilla.org
* [Ollama](https://ollama.com/) - A local LLM inference engine for running Llama 3, Mistral, Gemma, and other LLMs
* [aiortc](https://aiortc.readthedocs.io/en/latest/) - A Python Library for WebRTC and ORTC communication
* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).


