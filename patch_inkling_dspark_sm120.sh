#!/usr/bin/env bash
# patch-inkling-dspark-sm120.sh — patch the Inkling 2 Mini Docker image so the DSpark
# speculator runs on sm_120 (RTX PRO 6000 Blackwell).
#
# Three steps, run in order:
#   1. Swap in the official vLLM 0.34.0 core extension (_C_stable_libtorch.abi3.so),
#      which carries 69 sm_120 binaries including the reshape and cache kernels the
#      drafter's KV-cache write needs. The shipped image carries only sm_75 binaries.
#   2. Replace FlashInfer's precompiled RMSNorm in the drafter's dspark.py with plain
#      torch fp32 math — the precompiled cubins refuse to load on sm_120.
#   3. Restore the entrypoint to vllm serve — a prep container commits an image whose
#      ENTRYPOINT is sleep, which breaks the serve command.
#
# Usage: patch-inkling-dspark-sm120.sh <image> <tag> [wheel]
#   <image>   base image to patch, e.g. vllm/vllm-openai:inkling-2-mini-dspark
#   <tag>     tag for the patched image, e.g. vllm/vllm-openai:inkling-2-mini-dspark-fa120
#   [wheel]   path to the official vLLM 0.34.0 wheel; when omitted, downloads it from PyPI

set -euo pipefail

IMAGE="$1"
TAG="$2"
WHEEL="${3:-}"

if [ -z "$WHEEL" ]; then
    echo "Downloading the official vLLM 0.34.0 wheel from PyPI..."
    pip download --no-deps --only-binary :all: vllm==0.34.0 -d /tmp
    WHEEL=$(ls /tmp/vllm-0.34.0-*.whl | head -1)
fi

CONTAINER="inkling-dspark-sm120-patch-$$"

cleanup() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting throwaway container from $IMAGE ..."
docker run -d --name "$CONTAINER" --entrypoint sleep "$IMAGE" infinity >/dev/null

echo "Copying the official vLLM wheel into the container ..."
docker cp "$WHEEL" "$CONTAINER:/tmp/vllm.whl"

echo "Patching the core extension and the drafter's RMSNorm ..."
docker exec -i "$CONTAINER" python3 - <<'PYEOF'
import os, zipfile
import vllm

pkg = os.path.dirname(vllm.__file__)

with zipfile.ZipFile('/tmp/vllm.whl') as z:
    data = z.read('vllm/_C_stable_libtorch.abi3.so')
open(f'{pkg}/_C_stable_libtorch.abi3.so', 'wb').write(data)

path = f'{pkg}/models/inkling2/nvidia/dspark.py'
src = open(path).read()

old = '''    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        from flashinfer.norm import fused_add_rmsnorm, rmsnorm

        if residual is None:
            return rmsnorm(x, self.weight, self.eps)
        # In place: residual += x; x = norm(residual), from the unrounded sum.
        fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x'''

new = '''    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor:
        # torch-native stand-in for flashinfer.norm (its cubins lack sm_120):
        # the sum is taken in fp32, rounded once into residual, and the norm
        # reads the unrounded fp32 sum, exactly as the flashinfer contract.
        if residual is None:
            xf = x.float()
        else:
            xf = x.float() + residual.float()
            residual.copy_(xf.to(residual.dtype))
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)'''

if 'flashinfer.norm' in src:
    assert old in src, f'forward block not found in {path}; file may already be patched'
    open(path, 'w').write(src.replace(old, new))
    print(f'patched {path}')
else:
    print(f'{path} is already torch-native; skipped norm patch')
PYEOF

echo "Committing the patched image as $TAG ..."
docker commit --change 'ENTRYPOINT ["vllm","serve"]' "$CONTAINER" "$TAG" >/dev/null

echo "Done. Patched image: $TAG"
