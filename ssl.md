# Let's Encrypt 证书自动化更新配置

使用 `--manual` 模式生成的证书需要手动更新，可以通过 `--manual-auth-hook` 和 DNS 提供商的 API，实现自动化更新。

---

## 2. 准备工作

### 获取 DNS 提供商的 API 凭据

根据您使用的 DNS 服务商，获取相应的 API Token 或 Key。以下是一些常见服务商的文档：

- **Cloudflare**: [API 文档](https://developers.cloudflare.com/api/)
- **阿里云**: [API 文档](https://help.aliyun.com/document_detail/29739.html)
- **腾讯云**: [API 文档](https://cloud.tencent.com/document/api/302/8513)

---

## 3. 创建认证脚本

### 示例脚本: 使用 Cloudflare API 自动添加 TXT 记录

**文件名：`auth-hook.sh`**
```bash
#!/bin/bash

# Cloudflare API 信息
CF_API_TOKEN="YOUR_API_TOKEN"
CF_ZONE_ID="YOUR_ZONE_ID"

# 获取 Certbot 提供的 TXT 记录
CERTBOT_DOMAIN=$1
CERTBOT_VALIDATION=$2

# 添加 TXT 记录 https://developers.cloudflare.com/api/resources/dns/subresources/records/methods/create/
curl -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records" \
     -H "Authorization: Bearer $CF_API_TOKEN" \
     -H "Content-Type: application/json" \
     --data '{
         "type": "TXT",
         "name": "_acme-challenge.'$CERTBOT_DOMAIN'",
         "content": "'$CERTBOT_VALIDATION'",
         "ttl": 120
     }'
```

## 4. 保存并使脚本可执行

执行以下命令，将认证脚本设置为可执行：
```bash
chmod +x auth-hook.sh
```

## 5.配置自动更新任务 (cron)
```
crontab -e
# 每月 1 日凌晨运行自动更新任务
0 0 1 * * certbot renew --manual-auth-hook /path/to/auth-hook.sh --config-dir ~/certbot/config --work-dir ~/certbot/work --logs-dir ~/certbot/logs >> ~/certbot/certbot.log 2>&1

# 验证更新是否成功
certbot renew --dry-run

```


## 6.配置自动重载 Nginx
```
certbot renew \
    --manual-auth-hook /path/to/auth-hook.sh \
    --deploy-hook "systemctl reload nginx" \
    --config-dir ~/certbot/config \
    --work-dir ~/certbot/work \
    --logs-dir ~/certbot/logs

```


**注意事项**
1. DNS 记录 TTL: 确保 DNS TXT 记录的 TTL 足够低（如 120 秒）。
2. 脚本安全性: 脚本中包含敏感信息，建议将脚本权限设置为仅当前用户可读：
```bash
chmod 600 /path/to/auth-hook.sh
```
3. 日志管理: 定期检查 Certbot 和 Cron 日志，确保自动更新正常运行。