"""ID-Forcing on LongLive.

    python longlive/idforcing.py --prompt "A cat ..."                          # 2 min, one clip
    python longlive/idforcing.py --prompt_file prompts/moviegenbench_128.txt --seconds 240

The generator is LongLive as released (Wan2.1-T2V-1.3B + its LoRA, chunks of 3 latent frames, a
12-frame attention window whose first chunk is a frame sink). Nothing in the model is changed;
ID-Forcing only decides what the KV cache holds while each chunk is denoised. On LongLive it is
applied from the first chunk on, with self-caching only:

    chunk 0      LongLive's own first chunk (it attends to itself only); its KV is the sink
    chunk 1      sink 0-2 | self 3-5
    chunk 2      sink 0-2 | P(1) 3-5 | self 6-8
    chunk n>=3   sink 0-2 | P(n-2) 3-5 | P(n-1) 6-8 | self 9-11          (RoPE frames)

P(k) is chunk k self-cached: its KV computed attending to itself only, at the frames it was
generated at. An entry moves to the next slot by rotating the temporal RoPE of its keys (values
carry none), so the window's positions never grow with the video and every chunk attends to at
most 12 frames however long the rollout runs.

Noise: as in our experiments, the tape for the first NATIVE_CHUNKS chunks (60 s, LongLive's
training length) is drawn right after seeding, and the tape for the remaining chunks is drawn
when chunk NATIVE_CHUNKS is reached. A clip therefore starts the same at any length.

Clips are written as <output_folder>/<prompt[:100]>-<seed>-0.mp4, the naming our VBench
scripts read. The computation, noise included, is the same as in our experiments; see the README
on bit-exact reproducibility.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                  # holds wan_models/, checkpoints/, prompts/

F = 3                    # latent frames per chunk
FSL = 1560               # tokens per latent frame (60 x 104 latent, 2 x 2 patches)
LATENT = (16, 60, 104)   # channels, height, width of one latent frame
NATIVE_CHUNKS = 80       # first noise tape: 80 chunks = 240 latent frames = 60 s


# ------------------------------------------------------------------------------- KV cache
def new_kv_cache(frames, device):
    return [{"k": torch.zeros([1, frames * FSL, 12, 128], dtype=torch.bfloat16, device=device),
             "v": torch.zeros([1, frames * FSL, 12, 128], dtype=torch.bfloat16, device=device),
             "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
             "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
            for _ in range(30)]


def set_indices(cache, local_end_frames, global_end_frames):
    """Where the next chunk is written in the buffer (local) and where it sits in RoPE (global).
    LongLive attends to buffer frames 0-2 (the sink) plus frames 3 up to local_end + the chunk
    (at most 12 frames in all), so this also fixes what is read."""
    for c in cache:
        c["local_end_index"].fill_(int(local_end_frames) * FSL)
        c["global_end_index"].fill_(int(global_end_frames) * FSL)


def place(cache, kv, at_frame):
    lo = int(at_frame) * FSL
    for c, (k, v) in zip(cache, kv):
        c["k"][:, lo:lo + k.shape[1]] = k
        c["v"][:, lo:lo + v.shape[1]] = v


def read(cache, at_frame, frames):
    lo, hi = int(at_frame) * FSL, (int(at_frame) + int(frames)) * FSL
    return [(c["k"][:, lo:hi].clone(), c["v"][:, lo:hi].clone()) for c in cache]


def rotate(kv, delta, freqs_t):
    """Move cached KV along the time axis by `delta` latent frames. LongLive keeps keys with the
    RoPE already applied (float64 complex, then cast) and rotations compose, so this multiplies
    the temporal components by the phase of `delta`. Values carry no RoPE."""
    if delta == 0:
        return [(k.clone(), v) for k, v in kv]
    rot = freqs_t[abs(delta)]
    rot = rot if delta > 0 else rot.conj()
    nt = rot.shape[0]
    out = []
    for k, v in kv:
        b, s, n, d = k.shape
        c = torch.view_as_complex(k.to(torch.float64).reshape(b, s, n, d // 2, 2))
        c = torch.cat([c[..., :nt] * rot, c[..., nt:]], dim=-1)
        out.append((torch.view_as_real(c).flatten(-2).to(k.dtype), v))
    return out


# ------------------------------------------------------------------------------- forwards
def denoise(pipeline, cond, noisy_input, start_frame, cache, device):
    """LongLive's four denoising steps for one chunk at RoPE frame `start_frame`."""
    dsl = pipeline.denoising_step_list
    denoised_pred = None
    for index, current_timestep in enumerate(dsl):
        timestep = torch.ones([1, F], device=device, dtype=torch.int64) * current_timestep
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input, conditional_dict=cond, timestep=timestep,
            kv_cache=cache, crossattn_cache=pipeline.crossattn_cache,
            current_start=start_frame * FSL)
        if index < len(dsl) - 1:
            next_timestep = dsl[index + 1]
            noisy_input = pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1), torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones([F], device=device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
    return denoised_pred


def cache_clean(pipeline, cond, latents, start_frame, cache, device):
    """One forward of a clean chunk at t = context_noise (0); it writes the chunk's KV."""
    timestep = torch.ones([1, F], device=device, dtype=torch.int64) * pipeline.denoising_step_list[-1]
    context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
    pipeline.generator(
        noisy_image_or_video=latents, conditional_dict=cond, timestep=context_timestep,
        kv_cache=cache, crossattn_cache=pipeline.crossattn_cache, current_start=start_frame * FSL)


def self_cache(pipeline, cond, latents, start_frame, scratch, device):
    """The KV of a clean chunk at RoPE frame `start_frame`, attending to itself only."""
    set_indices(scratch, 0, start_frame)
    cache_clean(pipeline, cond, latents, start_frame, scratch, device)
    return read(scratch, 0, F)


# ------------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(pipeline, cond, num_chunks, seed, device):
    from utils.misc import set_seed
    window = pipeline.local_attn_size                      # 12 frames
    assert window == 4 * F, window
    scratch = new_kv_cache(2 * F, device)
    c = 128 // 2
    freqs_t = pipeline.generator.model.freqs.to(device).split(
        [c - 2 * (c // 3), c // 3, c // 3], dim=1)[0]

    set_seed(seed)
    tape = torch.randn([1, NATIVE_CHUNKS * F, *LATENT], device=device, dtype=torch.bfloat16)

    # ---- chunk 0: LongLive as released, in its own cache --------------------------------------
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device,
                                  kv_cache_size_override=window * FSL)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline.generator.model.local_attn_size = window
    pipeline._set_all_modules_max_attention_size(window)
    den = denoise(pipeline, cond, tape[:, 0:F], 0, pipeline.kv_cache1, device)
    cache_clean(pipeline, cond, den, 0, pipeline.kv_cache1, device)
    sink = read(pipeline.kv_cache1, 0, F)                 # chunk 0, cached at 0-2, seeing itself
    pipeline.kv_cache1 = None
    chunks = [den]

    # ---- chunks 1..: ID-Forcing's window -------------------------------------------------------
    cache = new_kv_cache(window, device)
    live = []                                             # self-cached entries, oldest first
    for n in range(1, num_chunks):
        if n == NATIVE_CHUNKS:
            tape = torch.randn([1, (num_chunks - NATIVE_CHUNKS) * F, *LATENT], device=device,
                               dtype=torch.bfloat16)
        i = n if n < NATIVE_CHUNKS else n - NATIVE_CHUNKS
        noisy = tape[:, i * F:(i + 1) * F]

        live = live[-2:]                                  # the oldest entry leaves the window
        for j, e in enumerate(live):                      # the rest move one slot back
            if e["pos"] != F * (j + 1):
                e["kv"] = rotate(e["kv"], F * (j + 1) - e["pos"], freqs_t)
                e["pos"] = F * (j + 1)
        place(cache, sink, 0)
        for j, e in enumerate(live):
            place(cache, e["kv"], F * (j + 1))
        self_pos = F * (len(live) + 1)                    # 3, 6, then 9 from chunk 3 on
        set_indices(cache, self_pos, self_pos)
        den = denoise(pipeline, cond, noisy, self_pos, cache, device)
        live.append(dict(kv=self_cache(pipeline, cond, den, self_pos, scratch, device),
                         pos=self_pos))
        chunks.append(den)
        if (n + 1) % 20 == 0:
            print(f"    chunk {n + 1}/{num_chunks}", flush=True)

    del cache, scratch, live, sink, tape
    return torch.cat(chunks, dim=1)


# ------------------------------------------------------------------------------- I/O
def save_video(pipeline, latents, path, fps):
    """Decode chunk by chunk with the VAE's streaming cache, frames to uint8 on the CPU as they
    come out, and write through a temporary file so an interrupted run leaves no partial clip."""
    from torchvision.io import write_video
    pipeline.vae.model.clear_cache()
    frames = []
    for i in range(0, latents.shape[1], F):
        px = pipeline.vae.decode_to_pixel(latents[:, i:i + F], use_cache=True)
        v = (px.to("cpu", dtype=torch.float32) * 0.5 + 0.5).clamp(0, 1)
        frames.append((v[0].permute(0, 2, 3, 1) * 255).to(torch.uint8))
        del px, v
    pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()
    frames = torch.cat(frames, dim=0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_video(path + ".part.mp4", frames, fps=fps)
    os.replace(path + ".part.mp4", path)
    return frames.shape[0]


def build_pipeline(config_path, generator_ckpt, lora_ckpt, device):
    import peft
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    config = OmegaConf.load(config_path)
    config.distributed = False
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(generator_ckpt, map_location="cpu", weights_only=False)
    pipeline.generator.load_state_dict(state["generator"])     # LongLive ships use_ema: false
    del state
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model, model_name="generator", lora_config=config.adapter,
        is_main_process=True)
    lora = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
    peft.set_peft_model_state_dict(
        pipeline.generator.model,
        lora["generator_lora"] if isinstance(lora, dict) and "generator_lora" in lora else lora)
    del lora
    pipeline.is_lora_enabled = True
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    pipeline.text_encoder.to(device=device)
    assert pipeline.frame_seq_length == FSL and pipeline.num_frame_per_block == F
    return pipeline


def clip_name(prompt, seed):
    return f"{prompt[:100].replace(os.sep, '_')}-{seed}-0.mp4"


def main():
    ap = argparse.ArgumentParser(description="ID-Forcing on LongLive")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="a single prompt")
    src.add_argument("--prompt_file", help="one prompt per line")
    ap.add_argument("--output_folder", default=os.path.join(ROOT, "outputs", "longlive"))
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="video length; one chunk is 0.75 s (12 frames at 16 fps)")
    ap.add_argument("--num_chunks", type=int, default=None, help="overrides --seconds")
    ap.add_argument("--seed", type=int, default=1356145)
    ap.add_argument("--generator_ckpt",
                    default=os.path.join(ROOT, "checkpoints", "longlive", "models", "longlive_base.pt"))
    ap.add_argument("--lora_ckpt",
                    default=os.path.join(ROOT, "checkpoints", "longlive", "models", "lora.pt"))
    ap.add_argument("--config_path", default=os.path.join(HERE, "configs", "longlive_inference.yaml"))
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0, help="this process takes prompts i with "
                    "i %% num_shards == shard")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    args = ap.parse_args()

    # resolve every user path before moving into the folder that holds wan_models/
    for k in ("prompt_file", "output_folder", "generator_ckpt", "lora_ckpt", "config_path"):
        if getattr(args, k):
            setattr(args, k, os.path.abspath(getattr(args, k)))
    if args.prompt is not None:
        prompts = [" ".join(args.prompt.split())]
    else:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [p for p in (ln.rstrip("\n").rstrip() for ln in f) if p]
    num_chunks = args.num_chunks or max(1, round(args.seconds * 4 / 3))
    mine = [(i, p) for i, p in enumerate(prompts) if i % args.num_shards == args.shard]
    os.makedirs(args.output_folder, exist_ok=True)

    os.chdir(ROOT)
    sys.path.insert(0, HERE)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    print(f"[ID-Forcing / LongLive] shard {args.shard}/{args.num_shards}: {len(mine)} prompt(s), "
          f"{num_chunks} chunks ({(12 * num_chunks - 3) / args.fps:.1f} s), seed {args.seed}",
          flush=True)
    if not mine:
        return
    pipeline = build_pipeline(args.config_path, args.generator_ckpt, args.lora_ckpt, device)

    for i, prompt in mine:
        path = os.path.join(args.output_folder, clip_name(prompt, args.seed))
        if os.path.isfile(path) and os.path.getsize(path) > 1024 and not args.overwrite:
            print(f"[{i}] exists, skipped: {path}", flush=True)
            continue
        t0 = time.time()
        cond = pipeline.text_encoder(text_prompts=[prompt])
        latents = rollout(pipeline, cond, num_chunks, args.seed, device)
        torch.cuda.empty_cache()
        n = save_video(pipeline, latents, path, args.fps)
        del latents, cond
        torch.cuda.empty_cache()
        print(f"[{i}] {n} frames in {(time.time() - t0) / 60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
