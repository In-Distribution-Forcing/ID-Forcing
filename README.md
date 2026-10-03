# ID-Forcing

ID-Forcing is an inference-time KV-cache scheme for long autoregressive video generation. It needs
no training: it runs on the released checkpoints of [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
and [LongLive](https://github.com/NVlabs/LongLive) without changing their weights or their model code.
It only decides what the KV cache holds while each chunk is denoised. Once ID-Forcing takes over,
every chunk attends to a fixed window of 12 latent frames at fixed RoPE positions, however long the
video runs.

Both models generate chunks of 3 latent frames (12 video frames, 0.75 s at 16 fps). Below,
*self-cached* means a chunk's KV is computed by a clean forward pass in which the chunk attends to
itself only. Cached entries move from slot to slot by rotating the temporal RoPE of their keys.

**Self-Forcing.** Chunks 0–6 (21 latent frames, the length Self-Forcing is trained on) come from
Self-Forcing's own rollout. From chunk 7 on, chunk *n* attends to:

| RoPE frames | 0–2 | 3–5 | 6–8 | 9–11 |
|---|---|---|---|---|
| block | sink: chunk 0 | chunk *n*−2, self-cached | chunk *n*−1, re-encoded attending to chunk *n*−2's self-cached KV | chunk *n* |

The block at frames 6–8 is rebuilt for every chunk and is never stored. After it is denoised,
chunk *n* is self-cached.

**LongLive.** ID-Forcing is applied from the first chunk on, with self-caching only. Chunk 0 is
LongLive's own first chunk and becomes the sink. Chunk 1 attends to `sink | self`, chunk 2 to
`sink | chunk 1 | self`, and every chunk *n* ≥ 3 to:

| RoPE frames | 0–2 | 3–5 | 6–8 | 9–11 |
|---|---|---|---|---|
| block | sink: chunk 0 | chunk *n*−2, self-cached | chunk *n*−1, self-cached | chunk *n* |

## Setup

```bash
conda create -n idforcing python=3.10 -y
conda activate idforcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Download the weights into the repository root:

```bash
# Wan2.1 base model (text encoder, VAE, config): used by both
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B

# Self-Forcing (DMD). ID-Forcing uses the EMA weights (`generator_ema`) of this checkpoint.
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

# LongLive-1.3B: base model + LoRA
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --include "models/*" --local-dir checkpoints/longlive
```

Checkpoint links:

- Self-Forcing: [gdhe17/Self-Forcing › checkpoints/self_forcing_dmd.pt](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt) (the EMA weights are used)
- LongLive: [Efficient-Large-Model/LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B) (`models/longlive_base.pt`, `models/lora.pt`)
- Wan2.1: [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

Expected layout:

```
wan_models/Wan2.1-T2V-1.3B/
checkpoints/self_forcing_dmd.pt
checkpoints/longlive/models/longlive_base.pt
checkpoints/longlive/models/lora.pt
```

## Generate

Both scripts make 2-minute videos by default. Give them a prompt with `PROMPT`, or a file with one
prompt per line with `PROMPT_FILE`:

```bash
PROMPT="..." bash run_self_forcing.sh                                # ID-Forcing on Self-Forcing
PROMPT_FILE=prompts/moviegenbench_128.txt bash run_longlive.sh       # ID-Forcing on LongLive
```

Options are environment variables:

| variable | default | meaning |
|---|---|---|
| `PROMPT` | – | a single prompt; overrides `PROMPT_FILE` |
| `PROMPT_FILE` | – | one prompt per line |
| `DURATION` | `120` | video length in seconds (120 s = 160 chunks = 1917 frames) |
| `SEED` | `1356145` | random seed |
| `GPUS` | `$CUDA_VISIBLE_DEVICES`, else `0` | comma-separated GPU ids; prompts are split across them, one process per GPU |
| `OUTPUT` | `outputs/self_forcing`, `outputs/longlive` | output folder |
| `PYTHON` | `python` | interpreter |
| `CKPT` | `checkpoints/self_forcing_dmd.pt` | Self-Forcing checkpoint (`run_self_forcing.sh`) |
| `GEN_CKPT`, `LORA_CKPT` | `checkpoints/longlive/models/{longlive_base,lora}.pt` | LongLive checkpoints (`run_longlive.sh`) |

Examples:

```bash
PROMPT="A white and orange tabby cat happily darts through a dense garden ..." bash run_self_forcing.sh
PROMPT_FILE=prompts/moviegenbench_128.txt GPUS=0,1,2,3 bash run_longlive.sh
DURATION=240 SEED=333 OUTPUT=outputs/sf_240s bash run_self_forcing.sh
```

Or call the Python entry points directly (`--help` lists every option):

```bash
python self_forcing/idforcing.py --prompt "..." --seconds 120 --seed 1356145
python longlive/idforcing.py --prompt_file prompts/moviegenbench_128.txt --seconds 240
```

Each video is saved as `<output>/<first 100 characters of the prompt>-<seed>-0.mp4` at 16 fps.
Videos that already exist are skipped, so you can restart an interrupted run with the same command.
`prompts/moviegenbench_128.txt` is the 128-prompt set we used for evaluation.

In our runs (A100 / RTX A6000), a 2-minute video took about 6 minutes on one GPU, including VAE
decoding. Noise is drawn exactly as in our experiments, and run side by side with our experiment
code on the same GPU, the scripts compute the same videos. On the same machine and software,
the Self-Forcing script is bit-for-bit repeatable. LongLive's pipeline is not: its outputs vary
slightly from run to run, with our experiment code as well. Expect visually identical videos
rather than bit-identical ones across different GPUs, drivers or library versions.

## Drift metrics

`eval/` measures how far a long video drifts away from its opening, on any folder of `.mp4` files.

- **Color shift** (`eval/color_shift.py`, CPU). The first and the last frame are converted to HSV,
  and the Hue channel of each is binned into an L1-normalised 180-bin histogram. We report the L1
  distance between the two histograms (0–2) and their Pearson correlation. Over the set we report
  ColorShift = 100 · (1 − mean L1 / 2); higher is better.
- **Motion drift** (`eval/motion_drift.py`, GPU). VBench's dynamic-degree test (RAFT optical flow
  on frames subsampled to 8 fps) is applied separately to the first and the last 5 seconds.
  LOST is the share of videos that move at the start but are static at the end, and
  LOST_score = 100 − LOST; higher is better. The script also reports the share moving at each end.

Motion drift needs VBench's `DynamicDegree` and the RAFT weights it downloads. Install them in an
environment of their own: VBench pins older packages, and we used Python 3.8 with torch 2.4.1.

```bash
pip install vbench==0.1.5 easydict opencv-python
mkdir -p ~/.cache/vbench/raft_model
wget -P ~/.cache/vbench/raft_model https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -d ~/.cache/vbench/raft_model ~/.cache/vbench/raft_model/models.zip   # -> models/raft-things.pth
```

Run both metrics on a folder of videos (`PYTHON` is the interpreter of that environment):

```bash
PYTHON=/path/to/vbench_env/bin/python bash eval/run_drift.sh outputs/self_forcing
PYTHON=... GPUS=0,1,2,3 bash eval/run_drift.sh outputs/longlive          # motion drift on 4 GPUs
PYTHON=... END=1917 bash eval/run_drift.sh outputs/sf_240s               # score 4 min clips at 2 min
```

The per-video results are written to `<folder>_color_shift.csv` and `<folder>_motion_drift.csv`,
and the summary is printed. Each script also runs on its own; see `--help`.

## Repository layout

```
run_self_forcing.sh, run_longlive.sh     entry scripts
self_forcing/idforcing.py                ID-Forcing rollout on Self-Forcing
self_forcing/configs/                    Self-Forcing inference config
self_forcing/{pipeline,utils,wan,demo_utils}/
longlive/idforcing.py                    ID-Forcing rollout on LongLive
longlive/configs/                        LongLive inference config (incl. LoRA settings)
longlive/{pipeline,utils,wan}/
eval/color_shift.py, eval/motion_drift.py, eval/run_drift.sh    drift metrics
prompts/
```

`self_forcing/` and `longlive/` contain only the files of the original repositories that inference
imports: Self-Forcing at commit `33593df` and LongLive at commit `e52d9ef`. These files are
unmodified. The only changes are to package `__init__.py` files: some were reduced to the modules
that are shipped, and empty ones were added to `utils/` and `demo_utils/`.

## Acknowledgements

This code builds on [Self-Forcing](https://github.com/guandeh17/Self-Forcing),
[LongLive](https://github.com/NVlabs/LongLive) and [Wan2.1](https://github.com/Wan-Video/Wan2.1).
The KV rotation in `self_forcing/idforcing.py` follows `_rope_time_delta_mul_` from Deep Forcing.
The code taken from Self-Forcing and LongLive keeps its original license (`self_forcing/LICENSE`,
`longlive/LICENSE`).
