# Live-Audio

Welcome to the Live-Audio repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules

``` sh

#0  source code

apt update
apt-get install build-essential libopenblas-dev vim  ffmpeg  git-lfs -y
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python


mkdir /asset
chmod 777 /asset/
git clone https://github.com/zwong91/Live-Audio.git
cd /workspace/Live-Audio
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

#2  Live-Audio
cd /workspace/Live-Audio
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

#3 xtts
cd /workspace/Live-Audio/src/xtts
pip install -e .[all,dev,notebooks]  -i https://pypi.tuna.tsinghua.edu.cn/simple

#4. download xtts-v2 
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download coqui/XTTS-v2  --local-dir  XTTS-v2

(rt) root@enty03:~/rt-audio# nvidia-smi
(rt) root@enty03:~/rt-audio# nvcc --version
(rt) root@enty03:~/rt-audio# pip show torch
```

## Q
***"`GLIBCXX_3.4.32' not found" error at runtime. GCC 13.2.0***
```
https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0
```

## Running with Docker

This will not guide you in detail on how to use CUDA in docker, see for
example [here](https://medium.com/@kevinsjy997/configure-docker-to-use-local-gpu-for-training-ml-models-70980168ec9b).

Still, these are the commands for Linux:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```

You can build the container image with:

```bash
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
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main --port 20000 --certfile cf.pem --keyfile cf.key --tts-type xtts-v2 --vad-type silero --vad-args '{"auth_token": "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"}' --llm-type llama
```

***test***

```
export PYANNOTE_AUTH_TOKEN=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
ASR_TYPE=sensevoice python -m unittest test.server.test_server
```
