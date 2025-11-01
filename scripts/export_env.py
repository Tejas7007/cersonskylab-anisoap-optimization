import json, platform
try:
    import torch
    torch_info = {
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as e:
    torch_info = {"error": str(e)}

out = {
  "python": platform.python_version(),
  **torch_info
}
print(json.dumps(out, indent=2))
