<#
.SYNOPSIS
    Create .venv-dml: a second environment that trains on an AMD or Intel GPU
    through DirectML.

.DESCRIPTION
    PyTorch's CUDA backend needs an NVIDIA card. torch-directml runs on any
    DirectX 12 GPU, but it is built against torch 2.4.1, older than the torch
    in uv.lock. So it lives in its own environment and the main one is left
    alone. Tests and everything CPU-only keep using `uv run`.

    Known limits of the DirectML backend: GRU layers are not supported (the
    CRNN must train on the CPU), and one AdamW operation falls back to the
    CPU with a warning.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_dml_env.ps1
    .venv-dml\Scripts\python.exe scripts\run_real.py --device dml ...
#>
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo '.venv-dml\Scripts\python.exe'

uv venv --python 3.12 (Join-Path $repo '.venv-dml')
uv pip install --python $py torch-directml numpy scipy soundfile scikit-learn lightgbm matplotlib
uv pip install --python $py --no-deps -e $repo
& $py -c "import torch, torch_directml; print('torch', torch.__version__, '| DirectML device:', torch_directml.device_name(0))"
