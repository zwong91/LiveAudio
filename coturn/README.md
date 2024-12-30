## 1. 测试
chrome://webrtc-internals
https://icetest.info
https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/

## 2. 引用

```
stun:108.136.246.72:3478

turn:108.136.246.72:3478
admin/admin
```

## 3. 鉴权信息使用SHA1的机制进行加密

```bash
1. 需要使用openssl的工具对于原本明文的部分进行加密，
例如：
echo -n "admin:gtp.aleopool.cc:password1" | openssl dgst -sha1

计算出来之后，在配置文件中：
# 启用 SHA1 认证机制
sha1-auth-enabled
# --- 用户凭证 ---
# 使用 SHA1 哈希后的密码
# 格式：user=username:SHA1(username:realm:password)
# 注意：您需要使用工具生成 SHA1 哈希值
user=admin:d937d8a8e499da7e2edafd045a618175117a2956



2. 客户端的代码也需要同步更改
const config = {
    iceServers: [
      { urls: 'stun:108.136.246.72:3478:3478' },
      { 
        urls: 'turn:108.136.246.72:3478:3478',
        username: 'admin',
        credential: 'd937d8a8e499da7e2edafd045a618175117a2956',
        credentialType: 'password'
      }
    ]
  };
  // 创建RTCPeerConnection
  function createPeerConnection() {
    const pc = new RTCPeerConnection(config);
    return pc;
  }

```