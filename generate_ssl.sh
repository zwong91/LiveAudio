#!/bin/sh

# 检查是否安装了 certbot
if ! command -v certbot &> /dev/null
then
    echo "certbot 未安装，正在安装..."
    apt update
    apt install -y certbot
else
    echo "certbot 已安装"
fi

/home/ubuntu/miniconda3/envs/rt/bin/python /home/ubuntu/front/rt-audio/generate_ssl_certificates.py
exec "$@"
