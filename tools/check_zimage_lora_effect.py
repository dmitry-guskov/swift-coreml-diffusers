#!/usr/bin/env python3
"""
Generate Z-Image outputs with and without a LoRA adapter and save side-by-side
artifacts so the LoRA effect can be checked quickly.

Backends
--------
- pytorch: works on Linux/macOS, uses the original Diffusers model and merges
  the LoRA weights into the transformer before sampling.
- coreml: works only on macOS, runs the exported 7-stage Core ML transformer
  plus Core ML VAE decoder.

Examples
--------
Linux / PyTorch:
python tools/check_zimage_lora_effect.py \
  --backend pytorch \
  --embeddings /path/to/zimage_embeddings.bin \
  --lora_safetensors /path/to/z_image_lora.safetensors \
  --out_dir /tmp/zimage_lora_check

macOS / Core ML:
python tools/check_zimage_lora_effect.py \
  --backend coreml \
  --transformer_dir /path/to/zimage_true_stage_fp32_7stage_lora_as_input_activation_space \
  --vae_coreml /path/to/VAEDecoder.mlmodelc \
  --embeddings /path/to/zimage_embeddings.bin \
  --lora_safetensors /path/to/z_image_lora.safetensors \
  --out_dir /tmp/zimage_lora_check
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from safetensors import safe_open

try:
    import coremltools as ct
except Exception:
    ct = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
TOTAL_BLOCKS = 30
BLOCKS_PER_STAGE = 5
BLOCK_STAGE_COUNT = TOTAL_BLOCKS // BLOCKS_PER_STAGE
STAGE_COUNT = 1 + BLOCK_STAGE_COUNT

BATCH = 1
IN_CHANNELS = 16
LATENT_H = 64
LATENT_W = 64
CAP_LEN = 77
CAP_DIM = 2560
PATCH_SIZE = 2
F_PATCH_SIZE = 1

DTYPE = torch.float32
SEQ_MULTI_OF = 32
TRAIN_STEP_COUNT = 1000
VAE_SCALING_FACTOR = 0.3611
VAE_SHIFT_FACTOR = 0.1159

LORA_RANK = 32
LORA_TARGETS = [
    ("adaLN_modulation.0", (LORA_RANK, 256), (15360, LORA_RANK)),
    ("attention.to_k", (LORA_RANK, 3840), (3840, LORA_RANK)),
    ("attention.to_out.0", (LORA_RANK, 3840), (3840, LORA_RANK)),
    ("attention.to_q", (LORA_RANK, 3840), (3840, LORA_RANK)),
    ("attention.to_v", (LORA_RANK, 3840), (3840, LORA_RANK)),
    ("feed_forward.w1", (LORA_RANK, 3840), (10240, LORA_RANK)),
    ("feed_forward.w2", (LORA_RANK, 10240), (3840, LORA_RANK)),
    ("feed_forward.w3", (LORA_RANK, 3840), (10240, LORA_RANK)),
]


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return int(n)


def stage_block_layer_indices(stage_id: int) -> List[int]:
    start = (stage_id - 1) * BLOCKS_PER_STAGE
    return list(range(start, start + BLOCKS_PER_STAGE))


def block_stage_ranges(block_stage_count: int, blocks_per_stage: int) -> List[Tuple[int, int]]:
    return [(i * blocks_per_stage, (i + 1) * blocks_per_stage - 1) for i in range(block_stage_count)]


def pad_sequence_runtime(sequences, batch_first=False, padding_value=0.0):
    if not isinstance(sequences, (list, tuple)) or len(sequences) == 0:
        raise ValueError("pad_sequence_runtime expects a non-empty list of tensors")
    max_len = max(int(s.shape[0]) for s in sequences)
    out = []
    for s in sequences:
        pad_len = max_len - int(s.shape[0])
        if pad_len > 0:
            pad = [0, 0] * (s.ndim - 1) + [0, pad_len]
            s = F.pad(s, pad, value=float(padding_value))
        out.append(s)
    dim = 0 if batch_first else 1
    return torch.stack(out, dim=dim)


class RopeEmbedderReal(nn.Module):
    def __init__(self, theta: float, axes_dims: List[int], axes_lens: List[int]):
        super().__init__()
        self.theta = float(theta)
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        for i, (d, e) in enumerate(zip(self.axes_dims, self.axes_lens)):
            assert d % 2 == 0
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d)))
            t = torch.arange(e, dtype=torch.float32)
            ang = torch.outer(t, freqs)
            pair = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
            self.register_buffer(f"freqs_pair_{i}", pair, persistent=True)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(torch.long)
        outs = []
        for i in range(len(self.axes_dims)):
            table = getattr(self, f"freqs_pair_{i}")
            outs.append(table.index_select(0, ids[:, i]))
        return torch.cat(outs, dim=1)


class ZSingleStreamAttnProcessorReal:
    _attention_backend = "native"
    _parallel_config = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
        **kwargs,
    ) -> torch.Tensor:
        del encoder_hidden_states, kwargs
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        bsz, seq_len, inner = q.shape
        heads = attn.heads
        head_dim = inner // heads

        q = q.view(bsz, seq_len, heads, head_dim)
        k = k.view(bsz, seq_len, heads, head_dim)
        v = v.view(bsz, seq_len, heads, head_dim)

        if getattr(attn, "norm_q", None) is not None:
            q = attn.norm_q(q)
        if getattr(attn, "norm_k", None) is not None:
            k = attn.norm_k(k)

        if freqs_cis is not None:
            if freqs_cis.ndim == 3:
                freqs_cis = freqs_cis.unsqueeze(0).expand(bsz, -1, -1, -1)
            cos = freqs_cis[..., 0].unsqueeze(2)
            sin = freqs_cis[..., 1].unsqueeze(2)

            def rope(x):
                x32 = x.float().reshape(bsz, seq_len, heads, -1, 2)
                x0 = x32[..., 0]
                x1 = x32[..., 1]
                y0 = x0 * cos - x1 * sin
                y1 = x0 * sin + x1 * cos
                return torch.stack([y0, y1], dim=-1).flatten(-2).to(dtype=x.dtype)

            q = rope(q)
            k = rope(k)

        attn_bias = None
        if attention_mask is not None:
            keep = attention_mask if attention_mask.dtype == torch.bool else (attention_mask > 0)
            if keep.ndim == 2:
                keep = keep[:, None, None, :]
            attn_bias = torch.logical_not(keep).to(q.dtype) * torch.tensor(-1e4, dtype=q.dtype, device=q.device)

        q_attn = q.transpose(1, 2).float()
        k_attn = k.transpose(1, 2).float()
        v_attn = v.transpose(1, 2).float()
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias.to(dtype=attn_scores.dtype)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v_attn).to(dtype=q.dtype).transpose(1, 2)
        out = out.reshape(bsz, seq_len, heads * head_dim)

        if isinstance(attn.to_out, (nn.ModuleList, list, tuple)):
            out = attn.to_out[0](out)
            if len(attn.to_out) > 1:
                out = attn.to_out[1](out)
        else:
            out = attn.to_out(out)

        if getattr(attn, "rescale_output_factor", None) is not None:
            out = out / attn.rescale_output_factor
        return out


def patch_rope(model):
    model.rope_embedder = RopeEmbedderReal(
        model.rope_embedder.theta,
        model.rope_embedder.axes_dims,
        model.rope_embedder.axes_lens,
    )
    proc = ZSingleStreamAttnProcessorReal()
    for module in model.modules():
        if isinstance(module, Attention):
            if hasattr(module, "set_processor"):
                try:
                    module.set_processor(proc)
                    continue
                except Exception:
                    pass
            module.processor = proc
    return model


def get_submodule_by_path(root: nn.Module, path: str) -> nn.Module:
    mod: nn.Module = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def load_lora_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def manually_merge_lora_into_weights(
    tr: nn.Module,
    lora_tensors: Dict[str, torch.Tensor],
    lora_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with torch.no_grad():
        for li in range(len(tr.layers)):
            layer = tr.layers[li]
            for target, A_shape, B_shape in LORA_TARGETS:
                a_key = f"diffusion_model.layers.{li}.{target}.lora_A.weight"
                b_key = f"diffusion_model.layers.{li}.{target}.lora_B.weight"
                A = lora_tensors[a_key].to(device=device, dtype=dtype)
                B = lora_tensors[b_key].to(device=device, dtype=dtype)
                if tuple(A.shape) != tuple(A_shape):
                    raise RuntimeError(f"LoRA A shape mismatch for {a_key}: {tuple(A.shape)} != {A_shape}")
                if tuple(B.shape) != tuple(B_shape):
                    raise RuntimeError(f"LoRA B shape mismatch for {b_key}: {tuple(B.shape)} != {B_shape}")
                mod = get_submodule_by_path(layer, target)
                if not isinstance(mod, nn.Linear):
                    raise TypeError(f"Expected nn.Linear at layers.{li}.{target}, got {type(mod)}")
                mod.weight.add_(lora_scale * (B @ A))


def load_embeddings(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    data = np.fromfile(path, dtype=np.float32)
    expected = BATCH * CAP_LEN * CAP_DIM
    if data.size != expected:
        raise RuntimeError(f"Embeddings tensor size mismatch: got {data.size}, expected {expected}")
    return torch.from_numpy(data.reshape(BATCH, CAP_LEN, CAP_DIM)).to(device=device, dtype=dtype)


def sample_initial_latents(seed: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    latents = torch.randn((BATCH, IN_CHANNELS, LATENT_H, LATENT_W), generator=g, dtype=dtype)
    return latents.to(device)


def shifted_sigmas(step_count: int, shift: float) -> List[float]:
    raw_timesteps = np.linspace(float(TRAIN_STEP_COUNT), 0.0, step_count + 1, dtype=np.float32)[:-1]
    raw_sigmas = raw_timesteps / float(TRAIN_STEP_COUNT)
    sigmas = [float(shift * s / (1.0 + (shift - 1.0) * s)) for s in raw_sigmas]
    sigmas.append(0.0)
    return sigmas


def timestep_schedule(step_count: int, shift: float) -> List[int]:
    sigmas = shifted_sigmas(step_count, shift)
    return [int(round(s * TRAIN_STEP_COUNT)) for s in sigmas[:-1]]


class TruePreambleWrapper(nn.Module):
    def __init__(self, tr: nn.Module, patch_size: int = 2, f_patch_size: int = 1):
        super().__init__()
        self.tr = tr
        self.patch_size = int(patch_size)
        self.f_patch_size = int(f_patch_size)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, cap_feats: torch.Tensor) -> torch.Tensor:
        bsz = latents.shape[0]
        x_list = [latents[i].unsqueeze(1) for i in range(bsz)]
        cap_list = [cap_feats[i] for i in range(bsz)]

        t = timestep * self.tr.t_scale
        t = self.tr.t_embedder(t)

        x, cap_feats_list, _x_size, x_pos_ids, cap_pos_ids, x_inner_pad_mask, cap_inner_pad_mask = self.tr.patchify_and_embed(
            x_list, cap_list, self.patch_size, self.f_patch_size
        )

        x_item_seqlens = [len(v) for v in x]
        x_max_item_seqlen = max(x_item_seqlens)
        x_cat = torch.cat(x, dim=0)
        x_cat = self.tr.all_x_embedder[f"{self.patch_size}-{self.f_patch_size}"](x_cat)
        adaln_input = t.type_as(x_cat)
        x_cat[torch.cat(x_inner_pad_mask)] = self.tr.x_pad_token
        x = list(x_cat.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.tr.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(v) for v in x_pos_ids], dim=0))
        x = pad_sequence_runtime(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence_runtime(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=latents.device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1
        for layer in self.tr.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        cap_item_seqlens = [len(v) for v in cap_feats_list]
        cap_max_item_seqlen = max(cap_item_seqlens)
        cap_cat = torch.cat(cap_feats_list, dim=0)
        cap_cat = self.tr.cap_embedder(cap_cat)
        cap_cat[torch.cat(cap_inner_pad_mask)] = self.tr.cap_pad_token
        cap_feats_list = list(cap_cat.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.tr.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(v) for v in cap_pos_ids], dim=0))
        cap_feats_list = pad_sequence_runtime(cap_feats_list, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence_runtime(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats_list.shape[1]]
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=latents.device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1
        for layer in self.tr.context_refiner:
            cap_feats_list = layer(cap_feats_list, cap_attn_mask, cap_freqs_cis)

        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats_list[i][:cap_len]], dim=0))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]], dim=0))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        unified_max_item_seqlen = max(unified_item_seqlens)
        unified = pad_sequence_runtime(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence_runtime(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=latents.device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
        return unified


class TrueStageMidWrapper(nn.Module):
    def __init__(self, tr: nn.Module, block_start: int, block_end: int, latent_h: int = 64, latent_w: int = 64, patch_size: int = 2, f_patch_size: int = 1):
        super().__init__()
        self.tr = tr
        self.block_start = int(block_start)
        self.block_end = int(block_end)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.patch_size = int(patch_size)
        self.f_patch_size = int(f_patch_size)

    def _build_unified_freq_and_mask(self, bsz: int, cap_feats: torch.Tensor, device):
        cap_ori_len = int(cap_feats.shape[1])
        cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
        cap_padded_len = cap_ori_len + cap_padding_len
        pH = pW = self.patch_size
        pF = self.f_patch_size
        f_tokens = 1 // pF
        h_tokens = self.latent_h // pH
        w_tokens = self.latent_w // pW
        x_len = f_tokens * h_tokens * w_tokens
        x_pos = self.tr.create_coordinate_grid(size=(f_tokens, h_tokens, w_tokens), start=(cap_padded_len + 1, 0, 0), device=device).flatten(0, 2)
        cap_pos = self.tr.create_coordinate_grid(size=(cap_padded_len, 1, 1), start=(1, 0, 0), device=device).flatten(0, 2)
        unified_pos = torch.cat([x_pos, cap_pos], dim=0)
        unified_freq = self.tr.rope_embedder(unified_pos).unsqueeze(0).expand(bsz, -1, -1, -1)
        unified_mask = torch.ones((bsz, x_len + cap_padded_len), dtype=torch.bool, device=device)
        return unified_freq, unified_mask

    def forward(self, hidden_tokens: torch.Tensor, timestep: torch.Tensor, cap_feats: torch.Tensor) -> torch.Tensor:
        adaln_input = self.tr.t_embedder(timestep * self.tr.t_scale).type_as(hidden_tokens)
        unified_freqs_cis, unified_attn_mask = self._build_unified_freq_and_mask(hidden_tokens.shape[0], cap_feats, hidden_tokens.device)
        x = hidden_tokens
        for li in range(self.block_start, self.block_end + 1):
            x = self.tr.layers[li](x, unified_attn_mask, unified_freqs_cis, adaln_input)
        return x


class TrueStageFinalWrapper(TrueStageMidWrapper):
    def forward(self, hidden_tokens: torch.Tensor, timestep: torch.Tensor, cap_feats: torch.Tensor) -> torch.Tensor:
        adaln_input = self.tr.t_embedder(timestep * self.tr.t_scale).type_as(hidden_tokens)
        unified_freqs_cis, unified_attn_mask = self._build_unified_freq_and_mask(hidden_tokens.shape[0], cap_feats, hidden_tokens.device)
        x = hidden_tokens
        for li in range(self.block_start, self.block_end + 1):
            x = self.tr.layers[li](x, unified_attn_mask, unified_freqs_cis, adaln_input)
        x = self.tr.all_final_layer[f"{self.patch_size}-{self.f_patch_size}"](x, adaln_input)
        x_list = list(x.unbind(dim=0))
        x_size = [(1, self.latent_h, self.latent_w) for _ in range(len(x_list))]
        x_list = self.tr.unpatchify(x_list, x_size, self.patch_size, self.f_patch_size)
        y = [v[:, 0, :, :] for v in x_list]
        return torch.stack(y, dim=0)


def build_stage_wrappers(tr: nn.Module):
    ranges = block_stage_ranges(BLOCK_STAGE_COUNT, BLOCKS_PER_STAGE)
    pre = TruePreambleWrapper(tr, PATCH_SIZE, F_PATCH_SIZE).eval()
    mids = [TrueStageMidWrapper(tr, start, end, LATENT_H, LATENT_W, PATCH_SIZE, F_PATCH_SIZE).eval() for start, end in ranges[:-1]]
    final = TrueStageFinalWrapper(tr, ranges[-1][0], ranges[-1][1], LATENT_H, LATENT_W, PATCH_SIZE, F_PATCH_SIZE).eval()
    return pre, mids, final


@torch.no_grad()
def run_dit_stagewise(tr: nn.Module, latents: torch.Tensor, time_step: int, cap_feats: torch.Tensor) -> torch.Tensor:
    pre, mids, final = build_stage_wrappers(tr)
    t = torch.tensor([float(TRAIN_STEP_COUNT - time_step) / float(TRAIN_STEP_COUNT)], device=latents.device, dtype=latents.dtype)
    hidden = pre(latents, t, cap_feats)
    for mid in mids:
        hidden = mid(hidden, t, cap_feats)
    return final(hidden, t, cap_feats)


@torch.no_grad()
def decode_latents_with_diffusers_vae(pipe: DiffusionPipeline, latents: torch.Tensor) -> Image.Image:
    unscaled = (latents / VAE_SCALING_FACTOR) + VAE_SHIFT_FACTOR
    decoded = pipe.vae.decode(unscaled).sample
    image = ((decoded / 2.0) + 0.5).clamp(0.0, 1.0)[0].permute(1, 2, 0).detach().cpu().numpy()
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_uint8)


@torch.no_grad()
def generate_with_pytorch(
    embeddings_path: Path,
    lora_path: Path,
    out_dir: Path,
    seed: int,
    step_count: int,
    scheduler_shift: float,
    lora_scale: float,
    device: torch.device,
    model_id: str,
) -> Dict[str, float]:
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=DTYPE).to(device)
    base_tr = patch_rope(pipe.transformer.to(device=device, dtype=DTYPE).eval())
    lora_tensors = load_lora_safetensors(lora_path)
    cap_feats = load_embeddings(embeddings_path, device=device, dtype=DTYPE)
    initial_latents = sample_initial_latents(seed, device, DTYPE)

    lora_tr = copy.deepcopy(base_tr).to(device=device, dtype=DTYPE).eval()
    manually_merge_lora_into_weights(lora_tr, lora_tensors, lora_scale=lora_scale, device=device, dtype=DTYPE)

    def sample(tr: nn.Module) -> torch.Tensor:
        latents = initial_latents.clone()
        sigmas = shifted_sigmas(step_count, scheduler_shift)
        for step_idx, time_step in enumerate(timestep_schedule(step_count, scheduler_shift)):
            model_out = run_dit_stagewise(tr, latents, time_step, cap_feats)
            velocity = -model_out
            dt = sigmas[step_idx + 1] - sigmas[step_idx]
            latents = latents + dt * velocity
        return latents

    no_lora_latents = sample(base_tr)
    with_lora_latents = sample(lora_tr)

    no_lora_image = decode_latents_with_diffusers_vae(pipe, no_lora_latents)
    with_lora_image = decode_latents_with_diffusers_vae(pipe, with_lora_latents)

    no_lora_path = out_dir / "no_lora.png"
    with_lora_path = out_dir / "with_lora.png"
    no_lora_image.save(no_lora_path)
    with_lora_image.save(with_lora_path)

    no_arr = np.asarray(no_lora_image, dtype=np.float32)
    with_arr = np.asarray(with_lora_image, dtype=np.float32)
    diff = np.abs(with_arr - no_arr)
    diff_img = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
    Image.fromarray(diff_img).save(out_dir / "abs_diff_x4.png")

    metrics = {
        "image_mae": float(diff.mean()),
        "image_max_abs": float(diff.max()),
        "latent_mae": float((with_lora_latents - no_lora_latents).abs().mean().item()),
        "latent_max_abs": float((with_lora_latents - no_lora_latents).abs().max().item()),
    }
    return metrics


def parse_compute_units(name: str):
    if ct is None:
        raise RuntimeError("coremltools is not installed")
    name = name.upper()
    if name == "CPU_ONLY":
        return ct.ComputeUnit.CPU_ONLY
    if name == "CPU_AND_GPU":
        return ct.ComputeUnit.CPU_AND_GPU
    if name == "ALL":
        return ct.ComputeUnit.ALL
    raise ValueError(f"Unsupported compute units: {name}")


def pack_stage_lora_vec_from_safetensors_np(lora_tensors: Dict[str, np.ndarray], stage_id: int) -> np.ndarray:
    flat: List[np.ndarray] = []
    for li in stage_block_layer_indices(stage_id):
        for target, A_shape, B_shape in LORA_TARGETS:
            a_key = f"diffusion_model.layers.{li}.{target}.lora_A.weight"
            b_key = f"diffusion_model.layers.{li}.{target}.lora_B.weight"
            A = lora_tensors[a_key]
            B = lora_tensors[b_key]
            if tuple(A.shape) != tuple(A_shape):
                raise RuntimeError(f"LoRA A shape mismatch for {a_key}: {tuple(A.shape)} != {A_shape}")
            if tuple(B.shape) != tuple(B_shape):
                raise RuntimeError(f"LoRA B shape mismatch for {b_key}: {tuple(B.shape)} != {B_shape}")
            flat.append(A.reshape(-1))
            flat.append(B.reshape(-1))
    return np.concatenate(flat, axis=0).astype(np.float32, copy=False)


def load_lora_safetensors_np(path: Path) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).cpu().numpy().astype(np.float32, copy=False)
    return tensors


def extract_single_output(pred: Dict[str, object]) -> np.ndarray:
    if len(pred) != 1:
        raise RuntimeError(f"Expected one output, got keys={list(pred.keys())}")
    value = next(iter(pred.values()))
    return np.array(value, dtype=np.float32, copy=False)


def ensure_coreml_runtime() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Core ML prediction is not available on Linux. "
            "Use --backend pytorch on the server, or run --backend coreml on macOS."
        )
    if ct is None:
        raise RuntimeError("coremltools is not installed")
    try:
        _ = ct.models.MLModel
    except Exception as exc:
        raise RuntimeError(f"coremltools runtime is unavailable: {exc}") from exc


def decode_latents_with_coreml_vae(vae_model, latents: np.ndarray) -> Image.Image:
    unscaled = (latents / VAE_SCALING_FACTOR) + VAE_SHIFT_FACTOR
    input_name = next(iter(vae_model.get_spec().description.input)).name
    pred = vae_model.predict({input_name: unscaled.astype(np.float32, copy=False)})
    image = extract_single_output(pred)
    if image.ndim == 4:
        image = image[0]
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image / 2.0) + 0.5, 0.0, 1.0)
    return Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))


def generate_with_coreml(
    transformer_dir: Path,
    vae_coreml_path: Path,
    embeddings_path: Path,
    lora_path: Path,
    out_dir: Path,
    seed: int,
    step_count: int,
    scheduler_shift: float,
    lora_scale: float,
    compute_units_name: str,
) -> Dict[str, float]:
    ensure_coreml_runtime()
    compute_units = parse_compute_units(compute_units_name)

    stage_models = [
        ct.models.MLModel(str(transformer_dir / f"ZImageTurbo_TransformerBackbone_stage{i}.mlpackage"), compute_units=compute_units)
        for i in range(STAGE_COUNT)
    ]
    vae_model = ct.models.MLModel(str(vae_coreml_path), compute_units=compute_units)
    lora_tensors = load_lora_safetensors_np(lora_path)
    cap_feats = np.fromfile(embeddings_path, dtype=np.float32).reshape(BATCH, CAP_LEN, CAP_DIM)
    rng = np.random.default_rng(seed)
    initial_latents = rng.standard_normal((BATCH, IN_CHANNELS, LATENT_H, LATENT_W), dtype=np.float32)
    zero_vec = np.zeros((sum(_numel(a) + _numel(b) for _, a, b in LORA_TARGETS) * BLOCKS_PER_STAGE,), dtype=np.float32)
    zero_scale = np.zeros((1,), dtype=np.float32)
    active_scale = np.array([lora_scale], dtype=np.float32)

    def sample(use_lora: bool) -> np.ndarray:
        latents = initial_latents.copy()
        sigmas = shifted_sigmas(step_count, scheduler_shift)
        time_steps = timestep_schedule(step_count, scheduler_shift)
        for step_idx, time_step in enumerate(time_steps):
            timestep = np.array([float(TRAIN_STEP_COUNT - time_step) / float(TRAIN_STEP_COUNT)], dtype=np.float32)
            pred0 = stage_models[0].predict(
                {
                    "latents": latents,
                    "timestep": timestep,
                    "cap_feats": cap_feats,
                    "lora_vec": zero_vec,
                    "lora_scale": zero_scale,
                }
            )
            hidden = extract_single_output(pred0)
            for stage_id in range(1, STAGE_COUNT):
                lora_vec = pack_stage_lora_vec_from_safetensors_np(lora_tensors, stage_id) if use_lora else zero_vec
                lora_scale_arr = active_scale if use_lora else zero_scale
                pred = stage_models[stage_id].predict(
                    {
                        "hidden_tokens": hidden,
                        "timestep": timestep,
                        "cap_feats": cap_feats,
                        "lora_vec": lora_vec,
                        "lora_scale": lora_scale_arr,
                    }
                )
                hidden = extract_single_output(pred)
            velocity = -hidden
            dt = sigmas[step_idx + 1] - sigmas[step_idx]
            latents = latents + dt * velocity
        return latents

    no_lora_latents = sample(use_lora=False)
    with_lora_latents = sample(use_lora=True)

    no_lora_image = decode_latents_with_coreml_vae(vae_model, no_lora_latents)
    with_lora_image = decode_latents_with_coreml_vae(vae_model, with_lora_latents)
    no_lora_image.save(out_dir / "no_lora.png")
    with_lora_image.save(out_dir / "with_lora.png")

    no_arr = np.asarray(no_lora_image, dtype=np.float32)
    with_arr = np.asarray(with_lora_image, dtype=np.float32)
    diff = np.abs(with_arr - no_arr)
    Image.fromarray(np.clip(diff * 4.0, 0, 255).astype(np.uint8)).save(out_dir / "abs_diff_x4.png")
    return {
        "image_mae": float(diff.mean()),
        "image_max_abs": float(diff.max()),
        "latent_mae": float(np.abs(with_lora_latents - no_lora_latents).mean()),
        "latent_max_abs": float(np.abs(with_lora_latents - no_lora_latents).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "pytorch", "coreml"], default="auto")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--transformer_dir", type=Path, default=None)
    ap.add_argument("--vae_coreml", type=Path, default=None)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--lora_safetensors", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("zimage_lora_effect_check"))
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler_shift", type=float, default=3.0)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--compute_units", choices=["CPU_ONLY", "CPU_AND_GPU", "ALL"], default="CPU_ONLY")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "auto":
        use_coreml = platform.system() == "Darwin" and args.transformer_dir is not None and args.vae_coreml is not None
        backend = "coreml" if use_coreml else "pytorch"
    else:
        backend = args.backend

    if backend == "coreml":
        if args.transformer_dir is None or args.vae_coreml is None:
            raise RuntimeError("--transformer_dir and --vae_coreml are required for --backend coreml")
        metrics = generate_with_coreml(
            transformer_dir=args.transformer_dir,
            vae_coreml_path=args.vae_coreml,
            embeddings_path=args.embeddings,
            lora_path=args.lora_safetensors,
            out_dir=args.out_dir,
            seed=args.seed,
            step_count=args.steps,
            scheduler_shift=args.scheduler_shift,
            lora_scale=args.lora_scale,
            compute_units_name=args.compute_units,
        )
    else:
        device = torch.device(args.device)
        metrics = generate_with_pytorch(
            embeddings_path=args.embeddings,
            lora_path=args.lora_safetensors,
            out_dir=args.out_dir,
            seed=args.seed,
            step_count=args.steps,
            scheduler_shift=args.scheduler_shift,
            lora_scale=args.lora_scale,
            device=device,
            model_id=args.model_id,
        )

    report = {
        "backend": backend,
        "seed": args.seed,
        "steps": args.steps,
        "scheduler_shift": args.scheduler_shift,
        "lora_scale": args.lora_scale,
        "embeddings": str(args.embeddings),
        "lora_safetensors": str(args.lora_safetensors),
        "metrics": metrics,
        "outputs": {
            "no_lora": str(args.out_dir / "no_lora.png"),
            "with_lora": str(args.out_dir / "with_lora.png"),
            "abs_diff_x4": str(args.out_dir / "abs_diff_x4.png"),
        },
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print(f"[DONE] backend={backend}")
    for key, value in metrics.items():
        print(f"[METRIC] {key}={value:.6f}")
    print(f"[DONE] saved {args.out_dir / 'no_lora.png'}")
    print(f"[DONE] saved {args.out_dir / 'with_lora.png'}")
    print(f"[DONE] saved {args.out_dir / 'abs_diff_x4.png'}")
    print(f"[DONE] saved {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
