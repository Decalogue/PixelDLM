from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Goose-World/rwkv-world-v3-subsample",
    repo_type="dataset",
    local_dir="./data/rwkv-world-v3-subsample",
    local_dir_use_symlinks=False
)