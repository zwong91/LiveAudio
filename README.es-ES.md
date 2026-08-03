# VoiceAgent

¡Bienvenido al repositorio de VoiceAgent! Este proyecto alberga aplicaciones emocionantes que aprovechan modelos avanzados de comprensión de audio y generación de habla para dar vida a tus experiencias de audio.

## Instalar

### Requisitos previos

```sh
#runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
# Clonar repositorio
git clone https://github.com/zwong91/VoiceAgent.git
cd VoiceAgent

# Dependencias del sistema (Ubuntu/Debian)
apt update
apt-get -qq -y install espeak-ng > /dev/null 2>&1
apt install curl ffmpeg libopenblas-dev vim git-lfs \
    build-essential cmake libasound-dev portaudio19-dev \
    libportaudio2 -y

# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

```

### Configuración del entorno

```sh
# Crear y activar entorno virtual de Python 3.10 llamado 'va'
uv venv --python=python3.10 va
source va/bin/activate

curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
pip install deepfilternet

# Instalar dependencias usando uv
uv pip install -r requirements.txt

# Instalar ollama https://github.com/ollama/ollama/releases
(curl -fsSL https://ollama.com/install.sh | sh && ollama serve > ollama.log 2>&1) &

ollama run gemma3:12b --verbose
```

### Abrir un túnel ngrok

Cuando se desarrolle y pruebe localmente, necesitarás abrir un túnel para redirigir las solicitudes a tu servidor de desarrollo local. Estas instrucciones utilizan ngrok.

Abre una terminal y ejecuta:

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | tee /etc/apt/sources.list.d/ngrok.list \
  && apt update \
  && apt install ngrok

ngrok config add-authtoken <token>

ngrok http 8765
```

Una vez que el túnel esté abierto, copia la URL de `Forwarding`. Se verá algo así como: `https://[tu-subdominio-ngrok].ngrok.app`. La necesitarás al configurar tu número de Twilio.

Ten en cuenta que el comando `ngrok` de arriba redirige a un servidor de desarrollo que se ejecuta en el puerto `5050`, que es el puerto predeterminado configurado en esta aplicación. Si anulas la variable `PORT` definida en `main.py`, necesitarás actualizar el comando `ngrok` en consecuencia.

Ten en cuenta que cada vez que ejecutes el comando `ngrok http`, se creará una nueva URL, y necesitarás actualizarla en todos los lugares donde se hace referencia a continuación.

## Configuración de Docker

1. Instalar NVIDIA Container Toolkit:

    Para usar GPU para el entrenamiento e inferencia de modelos en Docker, necesitas instalar NVIDIA Container Toolkit:

    Para usuarios de Ubuntu:

    ```bash
    # Agregar repositorio
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # Instalar nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # Reiniciar el servicio Docker
    sudo systemctl restart docker
    ```

    Para usuarios de otras distribuciones de Linux, consulta: [Guía de instalación de NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. Puedes construir la imagen del contenedor con:

    ```shell
    sudo docker build -t VoiceAgent .
    ```

    Después de obtener tu token VAD (ver las siguientes secciones) ejecuta:

    ```bash
    sudo docker volume create huggingface

    sudo docker run --gpus all -p 8765:8765 -v huggingface:/root/.cache/huggingface  -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' VoiceAgent
    ```

    El tema de los "volúmenes" te permitirá no volver a descargar los modelos de Hugging Face cada vez que vuelvas a ejecutar el contenedor. Si no lo necesitas, simplemente usa:

    ```bash
    sudo docker run --gpus all -p 8765:8765 -e PYANNOTE_AUTH_TOKEN='VAD_TOKEN_HERE' VoiceAgent
    ```

## Uso

**preparar**

[openai](https://platform.openai.com/) token de API.

[archivo pem](generate_ssl.sh) el micrófono necesita SSL/TLS

``` sh
# runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
HF_ENDPOINT=https://hf-mirror.com python3 -m src.main
```

***prueba***

```bash
export PYANNOTE_AUTH_TOKEN=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
ASR_TYPE=sensevoice python -m unittest test.server.test_server


```

## Preguntas frecuentes

1. Error "`GLIBCXX_3.4.32' not found" en tiempo de ejecución. GCC 13.2.0***

[https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0]

2. Cómo clonar una voz enviando el nombre de archivo de un archivo de onda que contiene la voz fuente

La clonación de voz funciona mejor con un archivo WAV mono de 22050 Hz de 16 bits que contiene una muestra corta (~5-30 seg) de la voz del hablante objetivo. La muestra debe ser una grabación limpia sin ruido de fondo ni música. El hablante debe hablar en un tono natural y conversacional. La muestra debe ser representativa de la voz del hablante, incluyendo su acento, entonación y estilo de habla.

3. Arquitectura de alto nivel de Coqui AI XTTS-v2 tts

XTTS utiliza un modelo VQ-VAE para discretizar el audio en tokens de audio.
Utiliza un modelo GPT para predecir estos tokens de audio según el texto de entrada y las variables latentes del hablante (speaker latents). Las variables latentes del hablante se calculan mediante una serie de capas de autoatención.
La salida del modelo GPT se pasa al modelo decodificador, que produce la señal de audio. Se utiliza un modelo de difusión para convertir la salida de GPT en fotogramas de espectrograma, y luego UnivNet genera la señal de audio final.

4. Kokoro TTS
Alternativa a Coqui AI XTTS-v2, Kokoro TTS es un sistema de texto a voz (TTS) que utiliza una combinación de redes neuronales y técnicas de aprendizaje profundo para generar habla de alta calidad a partir de texto de entrada. Está diseñado para producir un habla con sonido natural con un enfoque en la expresividad y la entonación emocional.
Kokoro TTS está construido sobre la arquitectura Tacotron, que es un modelo secuencia a secuencia que convierte texto en mel-espectrogramas. Los mel-espectrogramas luego se convierten en formas de onda de audio utilizando un vocoder, como WaveGlow o HiFi-GAN.


Recursos
---------

* [Documentación de WebRTC](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) - en <https://developer.mozilla.org>

* [Ollama](https://ollama.com/) - Un motor de inferencia LLM local para ejecutar Llama 3, Mistral, Gemma y otros LLM

* [aiortc](https://aiortc.readthedocs.io/en/latest/) - Una biblioteca de Python para comunicación WebRTC y ORTC
* [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) y [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).
