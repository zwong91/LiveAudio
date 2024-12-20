#!/bin/bash

# Cloudflare API 信息
CF_API_TOKEN="YOUR_API_TOKEN"
CF_ZONE_ID="YOUR_ZONE_ID"

# 获取 Certbot 提供的 TXT 记录
CERTBOT_DOMAIN=$1
CERTBOT_VALIDATION=$2

# 添加 TXT 记录
curl -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records" \
     -H "Authorization: Bearer $CF_API_TOKEN" \
     -H "Content-Type: application/json" \
     --data '{
         "type": "TXT",
         "name": "_acme-challenge.'$CERTBOT_DOMAIN'",
         "content": "'$CERTBOT_VALIDATION'",
         "ttl": 120
     }'
