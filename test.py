import httpx
import asyncio

async def upload_audio():
    url = 'https://xtts2.xyz666.org/generate_accent/vc_name_trump'
    headers = {
        'accept': 'application/json',
    }

    # 音频文件的 URL 地址
    file_urls = [
        'https://github.com/zwong91/rt-audio/raw/refs/heads/main/vc/liuyifei.wav'
    ]

    # 使用 httpx.AsyncClient 来异步发送请求，同时禁用 SSL 验证
    async with httpx.AsyncClient(verify=False) as client:
        # 准备上传的文件数据
        files = []
        for idx, file_url in enumerate(file_urls):
            # 下载文件内容，确保启用重定向
            response = await client.get(file_url, follow_redirects=True)  # Corrected here
            if response.status_code == 200:
                # 使用文件内容创建文件元组，并添加到 files 列表
                print(f"{response.content}")
                files.append(
                    ('files', (f'audio_{idx}.wav', response.content, 'audio/wav'))
                )
            else:
                print(f"Failed to download file from {file_url}, status code: {response.status_code}")
                return
        
        # 发送请求
        response = await client.post(url, headers=headers, files=files)

        if response.status_code == 200:
            print("Request successful!")
            print(response.json())  # 输出返回的 JSON 数据
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)

# 运行异步任务
asyncio.run(upload_audio())
