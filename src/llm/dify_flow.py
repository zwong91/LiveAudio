from .llm_interface import LLMInterface
from typing import List, Dict, AsyncGenerator
import os
import re
import asyncio
import aiohttp
import time
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)

req_host = 'http://23.249.20.24:5001'
req_url = '/enty_api/workflows-run/'
req_api_key = 'app-do7DxS7ro1gkWJZjmr47ORV5'
workflow_id = '8e780ed2-c7f7-4b0c-8718-18764b37e449'

class WorkflowLLM(LLMInterface):
    def __init__(self, model: str = "custom"):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + req_api_key
        }
        self.url = req_host + req_url + workflow_id

    async def generate_stream(self, messages: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        """流式生成回复，支持中断"""
        try:
            start_time = time.time()

            # 构造请求体
            data = {
                'inputs': {
                    'role': 'Emily Smith',
                    'user_id': '1234567890',  # 示例手机号
                    'in_chat_one_v1': query
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=self.headers, json=data) as response:
                    if response.status != 200:
                        logging.error(f"请求失败, 错误代码: {response.status}")
                        return

                    result = await response.json()
                    out_text = result['data']['outputs']['out_chat_one_v1']

                    # 处理文本并流式输出
                    words = out_text.split()
                    for word in words:
                        yield word + " "
                        await asyncio.sleep(0.05)  # 模拟流式输出

        except asyncio.CancelledError:
            print("Dify Flow generation cancelled")
            raise
        except Exception as e:
            logging.error(f"请求错误: {e}")
        finally:
            end_time = time.time()
            print(f"dify flow time: {end_time - start_time:.4f} seconds")
