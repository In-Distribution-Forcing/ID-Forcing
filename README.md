<p align="center">
<h1 align="center">ID-Forcing</h1>
<h3 align="center">In-Distribution Forcing for Long Video Generation at Test Time</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://swswss.github.io/">Jeongwoo Shin</a><sup>1*</sup>
    ·
    <a href="https://youngyoon911.github.io/">Youngyoon Choi</a><sup>1*</sup>
    ·
    <a href="https://github.com/jasonjo97">Sangwoo Jo</a><sup>2</sup>
    ·
    <a href="">Hyunmog Kim</a><sup>2</sup><br>
    <a href="https://aii.korea.ac.kr/people.html">Sungjoon Choi</a><sup>2†</sup>
    ·
    <a href="http://www.joonseok.net/home.html">Joonseok Lee</a><sup>1†</sup>
    ·
    <a href="https://sites.google.com/view/jaewoongchoi/home">Jaewoong Choi</a><sup>3†</sup>
    ·
    <a href="https://jaemoo-choi.github.io/">Jaemoo Choi</a><sup>4†</sup><br>
    <sup>1</sup>Seoul National University <sup>2</sup>Korea University <sup>3</sup>Sungkyunkwan University <sup>4</sup>Georgia Institute of Technology<br>
    <sup>*</sup>Equal contribution · <sup>†</sup>Corresponding author
  </p>
  <h3 align="center"><a href="">Paper</a> | <a href="https://in-distribution-forcing.github.io/">Website</a></h3>
</p>

---

## 💡 TL;DR

Beyond the training horizon, cached KV entries become out-of-distribution because of how they were
cached (the **KV-provenance problem**), and the video drifts. **ID-Forcing** keeps both KV caching
and KV conditioning in-distribution:

- **Self-Caching**: the earliest entries of the window are cached attending only to themselves,
  and the rest autoregressively on top of them.
- **Exact Rolling Window**: the first chunk κ<sub>0</sub> stays as a sink next to the *L*−1 most
  recent entries, re-rotated onto trained positions.
- **Training-Free**: runs on the released Self-Forcing and LongLive checkpoints, unchanged.

![KV operations of Self-Forcing vs. ID-Forcing](assets/method.png)

## TABLE OF CONTENTS
1. [Supported Base Models](#-supported-base-models)
2. [Requirements](#-requirements)
3. [Installation](#-installation)
4. [Quick Start](#-quick-start)
5. [Drift Metrics](#-drift-metrics)
6. [Repository Layout](#-repository-layout)
7. [Acknowledgements](#-acknowledgements)
8. [Citation](#-citation)
9. [License](#-license)

## 🤖 Supported Base Models

| Base Model | Checkpoint | LoRA | Weights | Window *L* | Self-cached *ℓ* | ID-Forcing takes over at |
|---|---|---|---|---|---|---|
| [Self-Forcing](https://github.com/guandeh17/Self-Forcing) | `self_forcing_dmd.pt` | — | EMA (`generator_ema`) | 3 | 2 | chunk 7, the first chunk past the 7-chunk (21-frame) training horizon |
| [LongLive](https://github.com/NVlabs/LongLive) | `longlive_base.pt` + `lora.pt` | ✅ | `generator` | 3 | 3 | chunk 1 (trained on minute-long videos) |

*L* counts the KV entries a chunk is conditioned on and *ℓ* the self-cached ones; both include the
sink κ<sub>0</sub>, as in the paper.

## 💻 Requirements

- One NVIDIA GPU with 40 GB+ VRAM (tested on A100 40 GB)
- Python 3.10, PyTorch ≥ 2.4 (tested with 2.5 and 2.6), flash-attn
- For Motion Drift only: a separate environment with VBench (see [Drift Metrics](#-drift-metrics))

## 📦 Installation

```bash
conda create -n idforcing python=3.10 -y
conda activate idforcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🚀 Quick Start

### 1. Download Checkpoints

Download the weights into the repository root:

```bash
# Wan2.1 base model (text encoder, VAE, config): used by both
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B

# Self-Forcing (DMD); ID-Forcing uses its EMA weights (`generator_ema`)
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .

# LongLive-1.3B: base model + LoRA
huggingface-cli download Efficient-Large-Model/LongLive-1.3B --include "models/*" --local-dir checkpoints/longlive
```

| Model | Source | Files |
|---|---|---|
| Self-Forcing | [gdhe17/Self-Forcing](https://huggingface.co/gdhe17/Self-Forcing/blob/main/checkpoints/self_forcing_dmd.pt) | `checkpoints/self_forcing_dmd.pt` (EMA weights used) |
| LongLive | [Efficient-Large-Model/LongLive-1.3B](https://huggingface.co/Efficient-Large-Model/LongLive-1.3B) | `models/longlive_base.pt`, `models/lora.pt` |
| Wan2.1 | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | text encoder, VAE, config |

Expected layout:

```
wan_models/Wan2.1-T2V-1.3B/
checkpoints/self_forcing_dmd.pt
checkpoints/longlive/models/longlive_base.pt
checkpoints/longlive/models/lora.pt
```

### 2. Run Inference

Both scripts make 2-minute videos by default. Pass a single prompt with `PROMPT`, or a file with
one prompt per line with `PROMPT_FILE`.

**With Self-Forcing:**
```bash
PROMPT="A white and orange tabby cat happily darts through a dense garden ..." bash run_self_forcing.sh
```

**With LongLive:**
```bash
PROMPT="A white and orange tabby cat happily darts through a dense garden ..." bash run_longlive.sh
```

**Many prompts on several GPUs, or a longer video:**
```bash
PROMPT_FILE=prompts/moviegenbench_128.txt GPUS=0,1,2,3 bash run_longlive.sh
DURATION=240 SEED=333 OUTPUT=outputs/sf_240s bash run_self_forcing.sh
```

Options are environment variables:

| Variable | Default | Meaning |
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

The Python entry points can also be called directly; `--help` lists every option:

```bash
python self_forcing/idforcing.py --prompt "..." --seconds 120 --seed 1356145
python longlive/idforcing.py --prompt_file prompts/moviegenbench_128.txt --seconds 240
```

Each video is saved as `<output>/<first 100 characters of the prompt>-<seed>-0.mp4` at 16 fps.
Videos that already exist are skipped, so you can restart an interrupted run with the same command.

## 📏 Drift Metrics

| Metric | Script | What it measures | Score (higher is better) |
|---|---|---|---|
| **Color Drift** | `eval/color_drift.py` (CPU) | 180-bin, L1-normalised HSV-hue histograms *h*<sub>start</sub>, *h*<sub>end</sub> of the first and the last frame (Xiang et al., 2026) | 100 · (1 − ½ ‖*h*<sub>start</sub> − *h*<sub>end</sub>‖<sub>1</sub>), averaged over videos |
| **Motion Drift** | `eval/motion_drift.py` (GPU) | VBench Dynamic Degree *d*<sub>start</sub>, *d*<sub>end</sub> ∈ {0, 1} of the first and the last 5 seconds (RAFT optical flow on frames subsampled to 8 fps) | 100 − LOST, where LOST is the % of videos with *d*<sub>start</sub> = 1 and *d*<sub>end</sub> = 0 |

```bash
# Motion Drift needs VBench and its RAFT weights (separate env; we used Python 3.8, torch 2.4.1)
pip install vbench==0.1.5 easydict opencv-python
mkdir -p ~/.cache/vbench/raft_model
wget -P ~/.cache/vbench/raft_model https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip -d ~/.cache/vbench/raft_model ~/.cache/vbench/raft_model/models.zip   # -> models/raft-things.pth

# both metrics on a folder of videos -> <folder>_color_drift.csv, <folder>_motion_drift.csv
PYTHON=/path/to/vbench_env/bin/python bash eval/run_drift.sh outputs/self_forcing
PYTHON=... GPUS=0,1,2,3 bash eval/run_drift.sh outputs/longlive          # Motion Drift on 4 GPUs
PYTHON=... END=1917 bash eval/run_drift.sh outputs/sf_240s               # score 4 min clips at 2 min
```

## 📂 Repository Layout

```
run_self_forcing.sh, run_longlive.sh     entry scripts
self_forcing/idforcing.py                ID-Forcing rollout on Self-Forcing
self_forcing/configs/                    Self-Forcing inference config
self_forcing/{pipeline,utils,wan,demo_utils}/
longlive/idforcing.py                    ID-Forcing rollout on LongLive
longlive/configs/                        LongLive inference config (incl. LoRA settings)
longlive/{pipeline,utils,wan}/
eval/                                    drift metrics (color_drift.py, motion_drift.py, run_drift.sh)
prompts/moviegenbench_128.txt            the 128 MovieGen evaluation prompts
```

`self_forcing/` and `longlive/` contain only the files of the original repositories that inference
imports: Self-Forcing at commit `33593df` and LongLive at commit `e52d9ef`. These files are
unmodified. The only changes are to package `__init__.py` files: some were reduced to the modules
that are shipped, and empty ones were added to `utils/` and `demo_utils/`.

## 🙏 Acknowledgements

This project builds upon the following works:

- [**Self-Forcing**](https://github.com/guandeh17/Self-Forcing): autoregressive video diffusion with self-forcing training
- [**LongLive**](https://github.com/NVlabs/LongLive): real-time interactive long video generation
- [**Wan2.1**](https://github.com/Wan-Video/Wan2.1): base video diffusion model
- [**Deep Forcing**](https://github.com/cvlab-kaist/DeepForcing): the KV rotation in `self_forcing/idforcing.py` follows its `_rope_time_delta_mul_`
- [**MovieGen**](https://ai.meta.com/research/movie-gen/): evaluation prompts. As in prior work, we use the first 128 prompts refined with Qwen2.5-7B-Instruct (the file is taken from [**MemRoPE**](https://github.com/YoungRaeKimm/MemRoPE))
- [**VBench**](https://github.com/Vchitect/VBench): Dynamic Degree, used by Motion Drift; Color Drift follows Xiang et al. (2026), *Pathwise Test-Time Correction for Autoregressive Long Video Generation*

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{idforcing2026,
  title   = {In-Distribution Forcing for Long Video Generation at Test Time},
  author  = {Shin, Jeongwoo and Choi, Youngyoon and Jo, Sangwoo and Kim, Hyunmog and Choi, Sungjoon and Lee, Joonseok and Choi, Jaewoong and Choi, Jaemoo},
  journal = {arXiv preprint arXiv:0000.00000},
  year    = {2026}
}
```

## 📝 License

The code taken from Self-Forcing and LongLive keeps its original Apache License 2.0
(`self_forcing/LICENSE`, `longlive/LICENSE`).
