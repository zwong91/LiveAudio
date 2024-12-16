import requests
import json

import logging

req_host = 'http://23.249.20.24:5001'
req_url = '/enty_api/workflows-run/'
req_api_key = 'app-do7DxS7ro1gkWJZjmr47ORV5'
workflow_id = '8e780ed2-c7f7-4b0c-8718-18764b37e449'
# workflow_id = '8884addf-61a6-435e-8844-3528a0fff1a5'


def chat_ai(chat_phone, chat_text):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + req_api_key
        }
        data = {
            'inputs': {
                'role': 'Emily Smith',
                'user_id': chat_phone,  # 手机号
                'in_chat_one_v1': chat_text  # 用户输入的文本
            }
        }
        url = req_host + req_url + workflow_id
        response = requests.post(url, headers=headers, json=data)

        # logging.info(f"chat api, url: {url}, headers: {headers}, data: {data}, response: {response.text} ...")
        if response.status_code != 200:
            logging.error(f"request failed, code != 200 , reason: {response.text} ...")
            return

        result_json = json.loads(response.text)
        return result_json['data']['outputs']['out_chat_one_v1']
    except Exception as e:
        logging.error(f"request failed, reason: {e} ...")


if __name__ == '__main__':
    logging.info('')

    phone = '13766058975'
    text = '今天几号?!'
    answer = chat_ai(phone, text)
    print(answer)
    logging.info(f"ai 聊天, 问: {text}, 答: {answer}")
