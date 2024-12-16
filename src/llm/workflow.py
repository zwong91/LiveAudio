from .llm_interface import LLMInterface
from typing import List, Optional, Tuple, Dict
import re
import os
import aiohttp
import asyncio
import json
from dotenv import load_dotenv
import logging

# 加载环境变量
load_dotenv(override=True)

req_host = 'http://23.249.20.24:5001'
req_url = '/enty_api/workflows-run/'
req_api_key = 'app-do7DxS7ro1gkWJZjmr47ORV5'
workflow_id = '8e780ed2-c7f7-4b0c-8718-18764b37e449'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class WorkflowLLM(LLMInterface):
    def __init__(self, model: str = "custom"):
        # 如果不需要做初始化，可以移除下面这行
        # openai.api_key = OPENAI_API_KEY  # OpenAI API key seems unused here
        pass  # 这里的 `pass` 只是为了避免缩进错误

    async def generate(self, history: List, query: str, max_tokens: int = 32) -> Tuple[str, List[Dict[str, str]]]:
        """根据对话历史生成回复"""

        if history is None:
            history = []

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + req_api_key
        }
        
        mobile = "1234567890"  # 示例手机号，实际中应从历史或用户输入中获取
        
        # 构造请求体
        data = {
            'inputs': {
                'role': 'Emily Smith',
                'user_id': mobile,  # 手机号
                'in_chat_one_v1': query  # 用户输入的文本
            }
        }

        url = req_host + req_url + workflow_id
        
        # 使用 aiohttp 异步请求
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        logging.error(f"请求失败, 错误代码: {response.status}, 原因: {await response.text()}")
                        return "服务器繁忙, 请稍后!", history  # 请求失败返回空字符串并保留历史

                    # 获取返回的数据
                    result_json = await response.json()
                    out_text = result_json['data']['outputs']['out_chat_one_v1']
                    
                    # 处理文本（例如：去除某些特定的标点符号）
                    processed_tts_text = ""
                    punctuation_pattern = r'([!?;。！？])'

                    escaped_processed_tts_text = re.escape(processed_tts_text)
                    tts_text = re.sub(f"^{escaped_processed_tts_text}", "", out_text)

                    logging.info(f"tts: {tts_text}")
                    # 更新对话历史
                    updated_history = history + [{'role': 'Emily Smith', 'text': out_text}]
                    return tts_text, updated_history

            except Exception as e:
                logging.error(f"请求错误: {e}")
                return "", history  # 如果发生异常，返回空字符串并保留历史
