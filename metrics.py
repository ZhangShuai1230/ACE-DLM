"""Defect metrics for continuous diffusion LM generations:
  - repetition : seq-rep-4 (Welleck et al. 2020), reported as median.
  - Gen-PPL    : under GPT-2 Large.
  - clean-PPL  : Gen-PPL of generations with seq-rep-4 <= ACCEPT_THRESHOLD (1.92%).
"""
from __future__ import annotations
import os
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Human 95th-percentile seq-rep-4. "clean" iff seq_rep_4 <= this.
ACCEPT_THRESHOLD = 0.0192

PPL_MODEL = "gpt2-large"


def seq_rep_4(text: str) -> float:
    """Fraction of repeated 4-grams (Welleck 2020). 0 = no repetition."""
    w = text.split()
    if len(w) < 5:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    c = Counter(grams)
    return sum(x - 1 for x in c.values() if x >= 2) / len(grams)


def summarize(texts) -> dict:
    """Corpus-level defect summary."""
    reps = sorted(seq_rep_4(t) for t in texts)
    return {
        "n": len(texts),
        "rep_median_pct": 100 * reps[len(reps) // 2],
        "rep_gt5pct_pct": 100 * sum(r > 0.05 for r in reps) / len(reps),
    }


def gen_ppl(texts, device, ppl_model=PPL_MODEL):
    """Gen-PPL + unigram entropy under a pretrained scorer (default GPT-2 Large)."""
    ne = [t for t in texts if t and t.strip()]
    ev = GenPPLEvaluator(ppl_model, batch_size=8, eval_context_size=1024, device=str(device))
    r = ev.evaluate(ne, max_length=1024)
    return r["ppl"], r["mean_entropy"]


def is_clean(text: str, threshold: float = ACCEPT_THRESHOLD) -> bool:
    """The compute-to-clean acceptance test: seq-rep-4 <= human 95th-pct (1.92%)."""
    return seq_rep_4(text) <= threshold


def accept_rate(texts, threshold: float = ACCEPT_THRESHOLD) -> float:
    """Fraction of generations passing the human repetition criterion."""
    return sum(is_clean(t, threshold) for t in texts) / max(len(texts), 1)


def clean_ppl_reject_to_n(generate_fn, device, n: int = 1000, cap: int = 10000,
                          threshold: float = ACCEPT_THRESHOLD, ppl_model=PPL_MODEL):
    """CANONICAL clean-PPL (reject-to-N): keep generating and rejecting samples whose
    seq-rep-4 exceeds the human bar until `n` are accepted, then return the Gen-PPL of
    those `n`.
    Returns dict: {clean_ppl, n_accepted, n_generated, accept_rate, censored, method}.
    """
    accepted, n_gen, batch = [], 0, 64
    while len(accepted) < n and n_gen < cap:
        texts = generate_fn(batch)
        for t in texts:
            if is_clean(t, threshold):
                accepted.append(t)
        n_gen += len(texts)
    censored = len(accepted) < n
    ppl = gen_ppl(accepted[:n], device, ppl_model)[0] if len(accepted) >= 50 else None
    return {
        "clean_ppl": ppl,
        "n_accepted": len(accepted),
        "n_generated": n_gen,
        "accept_rate": len(accepted) / max(n_gen, 1),
        "censored": censored,
        "method": "reject-to-n",
    }


class GenPPLEvaluator:
    def __init__(
        self,
        model_name_or_path: str = "gpt2-large",
        batch_size: int = 8,
        eval_context_size: int = 1024,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.batch_size = batch_size
        self.eval_context_size = eval_context_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
        ).to(self.device).eval()

    @torch.no_grad()
    def _compute_batch_nlls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        targets = input_ids[:, 1:]
        logits_pred = logits[:, :-1, :].float()
        log_normalizers = torch.logsumexp(logits_pred, dim=-1)
        target_logits = torch.gather(logits_pred, -1, targets.unsqueeze(-1)).squeeze(-1)
        nlls = log_normalizers - target_logits

        eos_id = self.tokenizer.eos_token_id
        is_eos = (input_ids == eos_id)
        first_eos = (torch.cumsum(is_eos.int(), dim=-1) == 1)
        token_mask = (input_ids != eos_id)
        valid = first_eos[:, 1:] | token_mask[:, 1:]
        return nlls, valid.to(nlls.dtype)

    def evaluate(
        self,
        text_samples: List[str],
        max_length: int = 1024,
    ) -> Dict:
        out = self.tokenizer(
            text_samples,
            return_tensors="pt",
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"]
        ctx = min(self.eval_context_size, input_ids.shape[1])

        n = input_ids.shape[0]
        per_sample_nll_sum = np.zeros(n, dtype=np.float64)
        per_sample_tok_cnt = np.zeros(n, dtype=np.float64)
        total_nll_sum = 0.0
        total_tok_sum = 0.0

        n_batches = (n + self.batch_size - 1) // self.batch_size
        for bi in tqdm(range(n_batches), desc="Eval-PPL"):
            s = bi * self.batch_size
            e = min(s + self.batch_size, n)
            ids = input_ids[s:e]
            att = attention_mask[s:e]
            for cs in range(0, ids.shape[1], ctx):
                ce = min(cs + ctx, ids.shape[1])
                ids_chunk = ids[:, cs:ce]
                att_chunk = att[:, cs:ce]
                if ids_chunk.shape[1] < 2:
                    continue
                nlls, valid = self._compute_batch_nlls(ids_chunk, att_chunk)
                weighted = (nlls * valid).cpu().numpy()
                v_np = valid.cpu().numpy()
                per_sample_nll_sum[s:e] += weighted.sum(axis=-1)
                per_sample_tok_cnt[s:e] += v_np.sum(axis=-1)
                total_nll_sum += float(weighted.sum())
                total_tok_sum += float(v_np.sum())

        if total_tok_sum > 0:
            ppl = float(np.exp(total_nll_sum / total_tok_sum))
        else:
            ppl = float("nan")

        with np.errstate(divide="ignore", invalid="ignore"):
            per_sample_ppl = np.exp(per_sample_nll_sum / np.maximum(per_sample_tok_cnt, 1.0))
        per_sample_ppl = np.where(per_sample_tok_cnt > 0, per_sample_ppl, np.nan).tolist()

        per_sample_entropy = []
        for i in range(n):
            valid_len = int(attention_mask[i].sum().item())
            valid_ids = input_ids[i, :valid_len].cpu().numpy()
            _, counts = np.unique(valid_ids, return_counts=True)
            probs = counts.astype(np.float32) / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            per_sample_entropy.append(entropy)

        mean_entropy = (
            sum(per_sample_entropy) / len(per_sample_entropy)
            if per_sample_entropy else float("nan")
        )
        return {
            "ppl": ppl,
            "per_sample_ppl": per_sample_ppl,
            "mean_entropy": mean_entropy,
            "per_sample_entropy": per_sample_entropy,
            "total_tokens": int(total_tok_sum),
            "num_samples": n,
        }
