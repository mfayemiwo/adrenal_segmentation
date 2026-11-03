#!/usr/bin/env bash
# Install nnU-Net v2 on a login node without replacing existing ROCm PyTorch.
# Run from the repository root: bash scripts/setup_nnunet_rocm.sh
# Do not run inside a queued GPU job or a compute node.
set -euo pipefail
cd "$(dirname "$0")/.."

module load amd-rocm/rocm-6.3.3
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0

if [[ ! -f .venv/bin/activate ]]; then
    echo "ERROR: .venv does not exist in repository root" >&2
    exit 2
fi
source .venv/bin/activate

echo "Python executable: $(command -v python)"
python -m pip --version

# Check ROCm identity before pip touches the environment. A login node
# may not expose a GPU, so do not test torch.cuda.is_available() here.
torch_before="$(python - <<'PY'
import json, torch
if not torch.version.hip:
    raise SystemExit("ERROR: installed PyTorch is not a ROCm build; refusing nnU-Net installation")
print(json.dumps({"version": torch.__version__, "hip": torch.version.hip}))
PY
)"
echo "Before installation: $torch_before"

# Pin the installed torch family exactly, including local (+rocm) version
# suffixes, to prevent pip resolving another CUDA/CPU build.
constraints="$(mktemp -t nnunet-rocm-constraints.XXXXXX)"
trap 'rm -f "$constraints"' EXIT
python - "$constraints" <<'PY'
import importlib.metadata as md
import sys
packages = ("torch", "torchvision", "torchaudio")
with open(sys.argv[1], "w") as f:
    for name in packages:
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            continue
        f.write(f"{name}=={version}\n")
        print(f"Protecting {name}=={version}")
PY

# The 2.8.1 release supports Python 3.12 and torch>=2.1.2,!=2.9.*.
# A resolver error is preferable to replacing the ROCm torch installation.
echo "Installing nnunetv2==2.8.1 with existing torch family protected..."
python -m pip install --constraint "$constraints" --upgrade-strategy only-if-needed "nnunetv2==2.8.1"

torch_after="$(python - <<'PY'
import json, torch
if not torch.version.hip:
    raise SystemExit("ERROR: no ROCm PyTorch after install")
print(json.dumps({"version": torch.__version__, "hip": torch.version.hip}))
PY
)"
if [[ "$torch_before" != "$torch_after" ]]; then
    echo "ERROR: torch build changed despite constraints! Before=$torch_before After=$torch_after" >&2
    exit 4
fi
echo "After installation (unchanged): $torch_after"

for cmd in nnUNetv2_plan_and_preprocess nnUNetv2_train nnUNetv2_predict; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: missing nnU-Net command: $cmd" >&2
        exit 3
    fi
    echo "Found $cmd: $(command -v "$cmd")"
    "$cmd" -h > /dev/null
done
python - <<'PY'
from importlib.metadata import version
import nnunetv2
print("nnU-Net installed:", version("nnunetv2"))
PY
echo "SUCCESS: CPU-side nnU-Net command imports verified. Now safe to resubmit the GPU job."
