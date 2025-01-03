import requests
import json

# 请求 URL
url = 'https://rtc.live.cloudflare.com/v1/turn/keys/{TURN_KEY}}/credentials/generate'

# 请求头
headers = {
    'Authorization': 'Bearer 92d1cf73915fe293f3402775db92d40b552dd6ea84babd32d17869733cb34e2b',
    'Content-Type': 'application/json',
}

# 请求体
data = {
    'ttl': 86400
}

# 发送 POST 请求
response = requests.post(url, headers=headers, json=data)

# 打印响应内容
if response.status_code == 201:
    response_data = response.json()
    
    # 获取 username 和 credential
    ice_servers = response_data.get('iceServers', {})
    username = ice_servers.get('username', None)
    credential = ice_servers.get('credential', None)
    
    print("Username:", username)
    print("Credential:", credential)
else:
    print(f"Request failed with status code {response.status_code}")
