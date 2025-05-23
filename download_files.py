from huggingface_hub import hf_hub_download, snapshot_download

from transformers import AutoTokenizer
import os

# 配置部分
HG_MODEL = "livekit/turn-detector"
MODEL_REVISION = "v0.2.0-intl"

# 使用 huggingface_hub 下载整个 repo（含 tokenizer, config 等）
snapshot_download(
    repo_id=HG_MODEL,
    revision=MODEL_REVISION,
    local_dir_use_symlinks=False,
)

local_path = hf_hub_download(
    repo_id="livekit/turn-detector",
    filename="onnx/model_q8.onnx",
    revision="v0.2.0-intl"
)

print("File downloaded to:", local_path)


conf_path = hf_hub_download(
    repo_id="livekit/turn-detector",
    filename="languages.json",
    revision="v0.2.0-intl"
)

print("Json File downloaded to:", conf_path)

print("Download complete!")
