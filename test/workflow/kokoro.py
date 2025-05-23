import requests
response = requests.get("http://localhost:8880/v1/audio/voices")
voices = response.json()["voices"]

# Example 1: Simple voice combination (50%/50% mix)
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "你好, 很高兴认识你!",
        "voice": "zf_xiaoni+af_nova",  # Equal weights
        "response_format": "mp3"
    }
)
# Save the .pt file
with open("zf_xiaoni+af_nova.pt", "wb") as f:
    f.write(response.content)

# Example 2: Weighted voice combination (67%/33% mix)
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "你好, 很高兴认识你!",
        "voice": "zm_yunxi(2)+af_sky(1)",  # 2:1 ratio = 67%/33%
        "response_format": "mp3"
    }
)

# Example 3: Download combined voice as .pt file
response = requests.post(
    "http://localhost:8880/v1/audio/voices/combine",
    json="zf_xiaoxiao(2)+af_xiaoyi(1)+af_nova(1)"  # 2:1 ratio = 67%/33%
)

# Save the .pt file
with open("zf_xiaoxiao(2)+af_xiaoyi(1)+af_nova(1).pt", "wb") as f:
    f.write(response.content)

# Use the downloaded voice file
response = requests.post(
    "http://localhost:8880/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "combined_voice",  # Use the saved voice file
        "response_format": "mp3"
    }
)
