# Low Perplexity is Repetition: A One-Dimensional Self-Conditioning Attractor in Continuous Diffusion LMs

[![arXiv](https://img.shields.io/badge/arXiv-2607.00588-b31b1b.svg)](https://arxiv.org/abs/2607.00588)

Continuous diffusion language models can score low perplexity by silently repeating
n-grams, a defect that perplexity rewards instead of penalizes. We trace this to a
one-dimensional attractor in the self-conditioning feedback loop and remove it with one cheap,
label-free steering direction `d`: subtract `lambda * d` from the fed-back clean estimate at
every sampling step. No retraining; the same `d` transfers across samplers and sizes.

This package ships the steering algorithm (`steering.py`) and the minimal code to run it.

## Install

```bash
# PyTorch matching your CUDA, then the rest of the deps
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Models

**ELF checkpoints** (B, M, L) into `models/`:

```bash
hf download embedded-language-flows/ELF-B-owt-torch --local-dir models/ELF-B-owt-torch
hf download embedded-language-flows/ELF-M-owt-torch --local-dir models/ELF-M-owt-torch
hf download embedded-language-flows/ELF-L-owt-torch --local-dir models/ELF-L-owt-torch
```

## Run

```bash
bash scripts/human_reference.sh    # derive the 1.92% human repetition bar from XSum
bash scripts/estimate.sh           # estimate d once on ELF-B -> direction_B.pt
bash scripts/sample.sh             # baseline vs steered for B, M, L, all using the same d_B
bash scripts/benchmark.sh          # compute-to-clean for B, M, L, all using the same d_B
```

Outputs:

```
results/human_reference.json     # 1.92% acceptance bar (XSum 95th-pct seq-rep-4)
results/direction_B.pt           # the steering direction d (estimated on ELF-B, reused for M/L)
results/sample_<size>.json       # baseline vs steered: rep, Gen-PPL
results/benchmark_<size>.json    # compute-to-clean: clean-PPL to N clean
```

## Use the steering in your own sampler

```python
from steering import build_elf, estimate_direction, generate, sampling_args
model = build_elf("B", "models/ELF-B-owt-torch", "cuda")
d = estimate_direction(model, n=400)          # one vector, reused everywhere
args = sampling_args(sc_cfg=3.0, noise=1.5)   # guidance + per-step noise; tune your sampler here
texts = generate(model, n=1000, steer_d=d, lam=2.0, args=args, steps=64)
```

## Acknowledgements

Built on [**ELF**](https://github.com/lillian039/ELF) (Embedded Language Flows), the base
continuous-diffusion LM: the `elf_pytorch/` package here is adapted from the official ELF
release, and all runs use its released checkpoints. Thanks to the ELF authors for
open-sourcing the model and code.

## Citation

```bibtex
@article{zhang2026repetition,
  title   = {Low Perplexity is Repetition: A One-Dimensional Self-Conditioning Attractor
             in Continuous Diffusion LMs},
  author  = {Zhang, Shuai and Chen, Zijie and He, Hongliang and Du, Lun and Lan, Zhenzhong},
  journal = {arXiv preprint arXiv:2607.00588},
  year    = {2026}
}
```
