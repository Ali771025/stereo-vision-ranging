"""Explicit inference device selection; GPU requests never silently use CPU."""
import re


def prepare_device(requested, torch_module=None):
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise ValueError('PyTorch is not installed in this Python environment') from exc
    torch = torch_module
    value = str(requested).strip().lower()
    if value == 'cpu':
        return torch.device('cpu'), {'device': 'cpu', 'torch': str(torch.__version__)}
    match = re.fullmatch(r'(?:cuda:)?(\d+)', value)
    if not match:
        raise ValueError('model.device must be cpu or one CUDA index, e.g. "0" or "cuda:0"')
    index = int(match.group(1))
    if not torch.cuda.is_available():
        raise ValueError('GPU requested but CUDA is unavailable. Install NVIDIA-compatible '
                         'PyTorch in the SAME Python environment; run the gpu command')
    if index >= torch.cuda.device_count():
        raise ValueError('Requested CUDA index does not exist in the visible GPUs')
    device = torch.device(f'cuda:{index}')
    try:
        # Availability alone does not prove driver/kernel compatibility.
        x = torch.ones((32, 32), device=device)
        value = float((x @ x)[0, 0].item())
        torch.cuda.synchronize(device)
        if value != 32.0:
            raise RuntimeError('unexpected CUDA calculation')
    except Exception as exc:
        raise ValueError('CUDA calculation failed; verify the GPU driver and PyTorch build') from exc
    return device, {'device': str(device), 'torch': str(torch.__version__),
                    'torch_cuda': str(torch.version.cuda),
                    'gpu_name': torch.cuda.get_device_name(index), 'cuda_calculation': 'passed'}


def check_gpu(cfg):
    import json
    requested = cfg['model']['device']
    if str(requested).lower() == 'cpu':
        raise ValueError('gpu check requires a CUDA device; use gpu --device 0')
    _, report = prepare_device(requested)
    print(json.dumps(report, indent=2))
