# Inkling 2 Mini DSpark speculator on sm_120

A script that patches the Inkling 2 Mini Docker image so the DSpark speculator runs on sm_120 (RTX PRO 6000 Blackwell).

The shipped image carries only sm_75 binaries for the drafter's KV-cache write, and the drafter's RMSNorm comes from FlashInfer's precompiled cubins, which refuse to load on sm_120. Both fail at launch. This script fixes both, plus a broken entrypoint.

## What the script does

1. Swaps in the official vLLM 0.34.0 core extension (`_C_stable_libtorch.abi3.so`), which carries 69 sm_120 binaries including the reshape and cache kernels.
2. Replaces FlashInfer's precompiled RMSNorm in the drafter's `dspark.py` with plain torch fp32 math — the same contract: with a residual present, the sum is computed in fp32, written rounded into the residual in place, and the norm reads the unrounded fp32 sum.
3. Restores the entrypoint to `vllm serve` — a prep container commits an image whose entrypoint is `sleep`, which breaks the serve command.

## Usage

```bash
./32_patch_inkling_dspark_sm120.sh <image> <tag> [wheel]
```

- `<image>` — the base image to patch, e.g. `vllm/vllm-openai:inkling-2-mini-dspark`
- `<tag>` — the tag for the patched image, e.g. `vllm/vllm-openai:inkling-2-mini-dspark-fa120`
- `[wheel]` — path to the official vLLM 0.34.0 wheel; when omitted, the script downloads it from PyPI

The patch is applied inside a throwaway container, and the patched image is committed under the tag you give. The throwaway container is removed on exit.

## Notes

- The official vLLM 0.34.0 wheel is required; the sm_120 binaries live only in that extension.
- A patched image can be tested with the chicken, franken and coding tests, and the DSpark speculator's acceptance length should be checked before you trust it.
