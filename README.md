# rt-audio

Welcome to the rt-audio repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules

``` sh

#0  source code

apt update
apt-get install vim  ffmpeg  git-lfs -y

git clone https://github.com/zwong91/rt-audio.git
cd /workspace/rt-audio
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

#2  rt-audio
cd /workspace/rt-audio
pip install -r requirements.txt

#3 xtts
cd /workspace/rt-audio/src/xtts
pip install -e .[all,dev,notebooks]  # Select the relevant extras

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
sudo docker build -t rt-audio .
```

After getting your VAD token (see next sections) run:

```bash
sudo docker volume create huggingface

sudo docker run --gpus all -p 8765:8765 -v huggingface:/root/.cache/huggingface  -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' rt-audio
```

The "volume" stuff will allow you not to re-download the huggingface models each
time you re-run the container. If you don't need this, just use:

```bash
sudo docker run --gpus all -p 19999:19999 -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' rt-audio
```

## Usage

**prepare**

[openai](https://platform.openai.com/) api token.

[pem file](https://blog.csdn.net/liuchenbaidu/article/details/136722001)

``` sh
python3 -m src.main --certfile cf.pem --keyfile cf.key --tts-type xtts-v2 --vad-type pyannote --vad-args '{"auth_token": "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"}'
```

***test***
```
export PYANNOTE_AUTH_TOKEN=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
ASR_TYPE=sensevoice python -m unittest test.server.test_server
```
