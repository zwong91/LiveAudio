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

python generate_ssl_certificates.py
exec "$@"


# manual https://juejin.cn/post/7205839782381928508
# ~/certbot/config/live/xyz666.org/
#    ssl_certificate fullchain.pem;
#    ssl_certificate_key privkey.pem;
certbot certonly \
    --manual \
    --manual-auth-hook auth-hook.sh \
    --preferred-challenges dns \
    -d xyz666.org \
    --config-dir ~/certbot/config \
    --work-dir ~/certbot/work \
    --logs-dir ~/certbot/logs

