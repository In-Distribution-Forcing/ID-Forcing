"""ID-Forcing on Self-Forcing.

    python self_forcing/idforcing.py --prompt "A cat ..."                      # 2 min, one clip
    python self_forcing/idforcing.py --prompt_file prompts/moviegenbench_128.txt --seconds 240

The generator is Self-Forcing's released DMD checkpoint with its EMA weights (Wan2.1-T2V-1.3B,
four denoising steps, chunks of 3 latent frames, a 21-frame KV cache). Nothing in the model is
changed; ID-Forcing only decides what the KV cache holds while each chunk is denoised.

Chunks 0-6 (21 latent frames, the length Self-Forcing is trained on) are Self-Forcing's own
rollout: denoise the chunk against the KV cache, then cache it with one clean forward at t = 0.

From chunk 7 on, chunk n attends to a fixed window of four 3-frame blocks:

    RoPE frames  0-2   sink   chunk 0's KV as Self-Forcing cached it (chunk 0 sees only itself)
                 3-5   P      chunk n-2, self-cached: its KV computed attending to itself only
                 6-8   C      chunk n-1, re-encoded attending to P -- one autoregressive step on
                              the previous chunk's self-cached KV. Rebuilt for every chunk and
                              never stored.
                 9-11  self   chunk n

Once denoised, chunk n is self-cached at frames 9-11. A self-cached entry moves to the next slot
by rotating the temporal RoPE of its keys (values carry none), so the window's positions never
grow with the video and every chunk attends to 12 frames however long the rollout runs.

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
NATIVE_CHUNKS = 7        # chunks 0-6 (21 latent frames) are Self-Forcing's own rollout
CACHE_FRAMES = 21        # Self-Forcing's KV cache


# ------------------------------------------------------------------------------- KV cache
def new_kv_cache(pipeline, frames, device):
    return [{"k": torch.zeros([1, frames * FSL, 12, 128], dtype=torch.bfloat16, device=device),
             "v": torch.zeros([1, frames * FSL, 12, 128], dtype=torch.bfloat16, device=device),
             "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
             "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
            for _ in range(pipeline.num_transformer_blocks)]


def set_indices(cache, local_end_frames, global_end_frames):
    """Where the next chunk is written in the buffer (local) and where it sits in RoPE (global).
    The model attends to buffer[0 : local_end + chunk], so this also fixes what is read."""
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


def rotate(pipeline, kv, delta_frames):
    """Move cached KV along the time axis by `delta_frames` latent frames.

    Keys are cached with RoPE already applied and rotations compose, so this multiplies the
    temporal channels of every key by the phase of `delta_frames` (in float64, then cast back,
    as Deep Forcing's `_rope_time_delta_mul_` does). Values carry no RoPE and pass through.
    """
    freqs = pipeline.generator.model.freqs
    out = []
    for k, v in kv:
        kk = k.clone()
        B, L, H, D = kk.shape
        c = D // 2
        t_c = c - 2 * (c // 3)                                    # temporal complex channels
        freqs_t, _, _ = freqs.to(kk.device).split([t_c, c // 3, c // 3], dim=1)
        shift = abs(int(delta_frames))
        assert 0 < shift < freqs_t.shape[0], delta_frames
        mult = freqs_t[shift] if delta_frames >= 0 else torch.conj(freqs_t[shift])
        mult = mult.view(1, 1, 1, t_c)
        time_ri = kk[..., : 2 * t_c]
        time_cx = torch.view_as_complex(time_ri.to(torch.float64).reshape(-1, t_c, 2))
        time_cx = time_cx * mult.to(time_cx.dtype)
        time_ri.copy_(torch.view_as_real(time_cx).reshape(B, L, H, t_c, 2).flatten(-2)
                      .to(time_ri.dtype))
        out.append((kk, v))
    return out


# ------------------------------------------------------------------------------- forwards
def denoise(pipeline, cond, noisy, start_frame, cache):
    """Self-Forcing's few-step sampler for one chunk at RoPE frame `start_frame`."""
    dsl = pipeline.denoising_step_list
    current_start = int(start_frame) * FSL
    denoised = None
    for i in range(len(dsl)):
        _, denoised = pipeline.generator(
            noisy_image_or_video=noisy, conditional_dict=cond,
            timestep=torch.ones([1, F], device=noisy.device, dtype=torch.int64) * dsl[i],
            kv_cache=cache, crossattn_cache=pipeline.crossattn_cache, current_start=current_start)
        if i < len(dsl) - 1:
            flat = denoised.flatten(0, 1)
            noisy = pipeline.scheduler.add_noise(
                flat, torch.randn_like(flat),
                dsl[i + 1] * torch.ones([F], device=noisy.device, dtype=torch.long)
            ).unflatten(0, denoised.shape[:2])
    return denoised


def cache_clean(pipeline, cond, latents, start_frame, cache):
    """One forward of a clean chunk at t = context_noise (0); it writes the chunk's KV."""
    pipeline.generator(
        noisy_image_or_video=latents, conditional_dict=cond,
        timestep=torch.ones([1, F], device=latents.device, dtype=torch.int64)
        * pipeline.args.context_noise,
        kv_cache=cache, crossattn_cache=pipeline.crossattn_cache,
        current_start=int(start_frame) * FSL)


def encode(pipeline, cond, latents, start_frame, scratch, prefix=None):
    """The KV of a clean chunk at RoPE frame `start_frame`, computed in a scratch cache.

    With no prefix the chunk attends to itself only (self-caching). With a prefix -- one chunk's
    KV, already rotated to the frames just before `start_frame` -- it attends to the prefix and
    itself.
    """
    n_pre = 0 if prefix is None else F
    assert n_pre + latents.shape[1] <= scratch[0]["k"].shape[1] // FSL, \
        "the scratch cache must hold the prefix and the chunk without rolling"
    set_indices(scratch, 0, 0)
    if prefix is not None:
        place(scratch, prefix, 0)
    set_indices(scratch, n_pre, start_frame)
    cache_clean(pipeline, cond, latents, start_frame, scratch)
    return read(scratch, n_pre, latents.shape[1])


# ------------------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout(pipeline, cond, num_chunks, seed, device):
    # One noise tape for the whole clip, then a separate seed for the sampler's re-noising.
    torch.manual_seed(seed)
    noise = torch.randn([1, num_chunks * F, *LATENT], device=device, dtype=torch.bfloat16)
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 1)

    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    cache = new_kv_cache(pipeline, CACHE_FRAMES, device)
    scratch = new_kv_cache(pipeline, 2 * F, device)
    out = torch.zeros([1, num_chunks * F, *LATENT], device=device, dtype=torch.bfloat16)
    sink, recent, live = None, [], []

    for n in range(num_chunks):
        noisy = noise[:, n * F:(n + 1) * F]

        # ---- chunks 0-6: Self-Forcing as released ----------------------------------------------
        if n < NATIVE_CHUNKS:
            den = denoise(pipeline, cond, noisy, n * F, cache)
            out[:, n * F:(n + 1) * F] = den
            cache_clean(pipeline, cond, den, n * F, cache)
            if n == 0:
                sink = read(cache, 0, F)          # chunk 0 attended to itself only
            recent = (recent + [den])[-2:]
            continue

        # ---- handover: the last two native chunks are self-cached at slots 0 and 1 ---------------
        if not live:
            live = [dict(latent=x, kv=encode(pipeline, cond, x, F * (j + 1), scratch), slot=j)
                    for j, x in enumerate(recent)]

        # ---- the oldest entry leaves; the other two move one slot back by rotation ----------------
        live = live[-2:]
        for j, e in enumerate(live):
            if e["slot"] != j:
                e["kv"] = rotate(pipeline, e["kv"], F * (j - e["slot"]))
                e["slot"] = j

        # ---- C: chunk n-1 re-encoded at frames 6-8, attending to chunk n-2's self-cached KV -------
        ctx = encode(pipeline, cond, live[1]["latent"], 2 * F, scratch, prefix=live[0]["kv"])

        # ---- denoise chunk n at frames 9-11 against  sink | P | C | self -----------------------
        place(cache, sink, 0)
        place(cache, live[0]["kv"], F)
        place(cache, ctx, 2 * F)
        set_indices(cache, 3 * F, 3 * F)
        den = denoise(pipeline, cond, noisy, 3 * F, cache)
        out[:, n * F:(n + 1) * F] = den

        # ---- self-cache chunk n where it was generated -----------------------------------------
        live.append(dict(latent=den.clone(), kv=encode(pipeline, cond, den, 3 * F, scratch),
                         slot=2))
        del ctx
        if (n + 1) % 20 == 0:
            print(f"    chunk {n + 1}/{num_chunks}", flush=True)

    del cache, scratch, live, sink, recent, noise
    return out


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


def build_pipeline(config_path, checkpoint_path, device):
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline
    config = OmegaConf.load(config_path)
    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "generator_ema" not in state:
        raise KeyError(f"{checkpoint_path} has no 'generator_ema'; ID-Forcing uses the EMA weights "
                       "of Self-Forcing's released checkpoint (checkpoints/self_forcing_dmd.pt)")
    pipeline.generator.load_state_dict(state["generator_ema"])
    del state
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    assert pipeline.frame_seq_length == FSL and pipeline.num_frame_per_block == F
    return pipeline


def clip_name(prompt, seed):
    return f"{prompt[:100].replace(os.sep, '_')}-{seed}-0.mp4"


def main():
    ap = argparse.ArgumentParser(description="ID-Forcing on Self-Forcing")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="a single prompt")
    src.add_argument("--prompt_file", help="one prompt per line")
    ap.add_argument("--output_folder", default=os.path.join(ROOT, "outputs", "self_forcing"))
    ap.add_argument("--seconds", type=float, default=120.0,
                    help="video length; one chunk is 0.75 s (12 frames at 16 fps)")
    ap.add_argument("--num_chunks", type=int, default=None, help="overrides --seconds")
    ap.add_argument("--seed", type=int, default=1356145)
    ap.add_argument("--checkpoint_path",
                    default=os.path.join(ROOT, "checkpoints", "self_forcing_dmd.pt"))
    ap.add_argument("--config_path", default=os.path.join(HERE, "configs", "self_forcing_dmd.yaml"))
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0, help="this process takes prompts i with "
                    "i %% num_shards == shard")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    args = ap.parse_args()

    # resolve every user path before moving into the folder that holds wan_models/
    for k in ("prompt_file", "output_folder", "checkpoint_path", "config_path"):
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
    print(f"[ID-Forcing / Self-Forcing] shard {args.shard}/{args.num_shards}: {len(mine)} prompt(s), "
          f"{num_chunks} chunks ({(12 * num_chunks - 3) / args.fps:.1f} s), seed {args.seed}",
          flush=True)
    if not mine:
        return
    pipeline = build_pipeline(args.config_path, args.checkpoint_path, device)

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
