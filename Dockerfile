# Use an NVIDIA CUDA base image with Python 3
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Avoid interactive prompts from apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install any needed packages
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install \
                   ffmpeg \
                   libsndfile1 \
                   python3-pip \
                   python${PYTHON_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file
COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Make port 19999 available to the world outside this container
EXPOSE 19999

# Define environment variable
ENV NAME=RTAudio

# Set the entrypoint to your application
ENTRYPOINT ["python3", "-m", "src.main"]

# python -m gunicorn --bind 0.0.0.0:444  --workers 1 --reload --timeout 0 --keyfile /etc/letsencrypt/live/${SSL_DOMAIN_NAME}/privkey.pem --certfile /etc/letsencrypt/live/${SSL_DOMAIN_NAME}/cert.pem src.main -k uvicorn.workers.UvicornWorker
# Provide a default command (can be overridden at runtime)
CMD ["--host", "0.0.0.0", "--port", "19999"]
