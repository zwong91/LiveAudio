import os
import logging

import http.client
from typing import List

from tqdm import tqdm
import boto3
import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

app = typer.Typer()


@app.command("get_url")
def get_url(file_name: str):
    bucket_url = os.getenv("S3_BUCKET_URL")
    url = f"{bucket_url}/{file_name}"
    logging.info(f"url:{url}")

    return url


@app.command("r2")
def r2_upload(remote_folder: str, local_file: str) -> str:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    access_key = os.getenv("CLOUDFLARE_ACCESS_KEY")
    secret_key = os.getenv("CLOUDFLARE_SECRET_KEY")
    region_name = os.getenv("CLOUDFLARE_REGION")
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    return upload(remote_folder, local_file, endpoint_url, access_key, secret_key, region_name)


@app.command("s3")
def upload(
    remote_folder: str,
    local_file: str,
    endpoint_url: str = "",
    access_key: str = "",
    secret_key: str = "",
    region_name: str = "",
) -> str:
    endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL")
    access_key = access_key or os.getenv("AWS_ACCESS_KEY")
    secret_key = secret_key or os.getenv("AWS_SECRET_KEY")
    region_name = region_name or os.getenv("AWS_REGION")
    s3 = boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name,  # Must be one of: wnam, enam, weur, eeur, apac, auto
    )
    file_name = os.path.basename(local_file)
    file_size = os.path.getsize(local_file)
    logging.info(f"Name: {file_name} Size: {round(file_size/1024,2)} KB")

    with tqdm(
        total=file_size, unit="B", unit_scale=True, desc=f"Uploading {file_name} to R2", ascii=True
    ) as pbar:
        with open(local_file, "rb") as f:
            s3.upload_fileobj(f, remote_folder, file_name, Callback=lambda x: pbar.update(x))
    object_information = s3.head_object(Bucket=remote_folder, Key=file_name)
    logging.info(f"Upload {local_file} ok,object_information: {object_information}")

    # Delete object
    # s3.delete_object(Bucket=remote_folder, Key=file_name)
    return get_url(file_name)


@app.command("d1_table_query")
def d1_table_query(db_id: str, sql: str, sql_params: List[str] = []) -> dict:
    """
    https://developers.cloudflare.com/api/operations/cloudflare-d1-query-database
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")

    payload = {
        "params": sql_params,
        "sql": sql,
    }
    body = json.dumps(payload)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn = http.client.HTTPSConnection("api.cloudflare.com")
    conn.request(
        "POST",
        f"/client/v4/accounts/{account_id}/d1/database/{db_id}/query",
        body,
        headers,
    )
    res = conn.getresponse()
    data = res.read().decode("utf-8")
    # print(data)
    json_data = json.loads(data)
    logging.info(f"body:{body}, db_id:{db_id}, query res:{json_data}")
    return json_data


@app.command("d1_db")
def d1_db(db_id: str) -> dict:
    """
    https://developers.cloudflare.com/api/operations/cloudflare-d1-get-database
    """
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_KEY")

    conn = http.client.HTTPSConnection("api.cloudflare.com")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    conn.request(
        "GET",
        f"/client/v4/accounts/{account_id}/d1/database/{db_id}",
        headers=headers,
    )

    res = conn.getresponse()
    data = res.read()

    data = res.read().decode("utf-8")
    json_data = json.loads(data)
    logging.info(f"get db_id:{db_id}, query res:{json_data}")
    return json_data


r"""
python -m cf_rest_api r2 podcast audios/podcast/LLM.mp3
python -m cf_rest_api img_url LLM.mp3

python -m cf_rest_api d1_db \
    09f7a7c7-66ae-41ea-9dbe-b8b635b19758

python -m cf_rest_api d1_table_query \
    09f7a7c7-66ae-41ea-9dbe-b8b635b19758 \
    "select * from podcast limit 1"
"""
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    app()
