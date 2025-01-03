import httpx
import asyncio

async def send_request():
    url = "https://gtp.aleopool.cc/cf-calls"  # 目标 FastAPI 服务地址
    push_url = "https://whip.xyz666.org/publish/my-live"  # 要传递的 push_url

    # 创建一个 HTTP 客户端实例，设置较长的超时时间
    timeout = httpx.Timeout(30.0)  # 设定超时时间为 30 秒
    async with httpx.AsyncClient(timeout=timeout) as client:
        # 将 push_url 作为查询参数传递
        response = await client.post(url, params={"push_url": push_url})
        
        if response.status_code == 200:
            print(f"Request succeeded: {response.json()}")
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
        
        
    # 请求完成后，等待 500000 秒
    print("Sleeping for 5 seconds...")
    await asyncio.sleep(5000000)

# 在一个 asyncio 环境中运行这个请求
asyncio.run(send_request())
