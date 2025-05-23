from huggingface_hub import hf_hub_download

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
