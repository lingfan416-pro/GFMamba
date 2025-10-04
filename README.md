# 🎭 GFMamba: Multimodal Sentiment Analysis with Mamba-based Fusion

A PyTorch-based implementation of **multimodal sentiment analysis**, using the **GFMamba** model that integrates **text**, **audio**, and **vision** features for tasks like CMU-MOSI and CMU-MOSEI.

---

## 🧠 Model Overview

**GFMamba** is a multimodal fusion model that leverages the Mamba architecture to integrate and learn from:

- `Text` (language content)
- `Audio` (voice tone & prosody)
- `Vision` (facial expressions & motion)

It performs **regression-based sentiment prediction**, aiming to predict a sentiment score per utterance or clip.
datasets use cmu-mosi, if you want to transfer other datasets, you can change the yaml and datasets floder

---

## 🛠️ Installation

### 🔹 Clone and install dependencies

```bash
git clone https://github.com/yourusername/gfmamba.git
cd gfmamba
pip install -r requirements.txt

## requirement

```bash
pip install pytorch>2.1 python>3.9

if you interested in this project,concat us by email zzhe232@aucklanduni.ac.nz

---

## 🚀 Quick Start (macOS & Windows)

The repository includes a simplified inference script that lets you run multimodal sentiment analysis on your own `.MOV` video and `.txt` transcript without any extra training. Share these instructions with teammates so they can reproduce the same workflow on macOS or Windows.

### 1. Prepare the environment

| Platform | Command |
| --- | --- |
| **macOS / Linux (Bash)** | `python3 -m venv .venv && source .venv/bin/activate` |
| **Windows (PowerShell)** | `python -m venv .venv; .\.venv\Scripts\Activate.ps1`

Then install dependencies inside the virtual environment:

```bash
pip install -r requirements.txt
```

> ℹ️ **Note:** The first run downloads the `bert-base-uncased` checkpoint from HuggingFace. Ensure the machine has internet access or pre-populate the cache (`~/.cache/huggingface` on macOS/Linux, `%USERPROFILE%\.cache\huggingface` on Windows).

### 2. Point to your input files

1. Place the `.txt` transcript and `.MOV` video anywhere on the machine.
2. Open `simplified_inference.py` and update the file paths near the bottom of the script:  
   ```python
   txt_path = "<absolute path to your transcript>"
   mov_path = "<absolute path to your MOV video>"
   ```
   Absolute paths work best (e.g., `C:\\Users\\alice\\Desktop\\sample.txt` on Windows or `/Users/alice/Desktop/sample.txt` on macOS).

### 3. Run the simplified inference

From the project root (`GFMamba-main`):

```bash
python3 simplified_inference.py   # macOS / Linux
```

```powershell
python simplified_inference.py    # Windows
```

The script prints:

- Device information (CPU/GPU)
- Extracted feature shapes for text, audio, and video
- Sentiment scores for each modality and the fused output
- The final label (`正面` / `中性` / `负面`)

It also writes the latest result to `simplified_result.json` in the repository root.

### 4. Optional: Use the full GFMamba model

If you want to use the full model instead of the heuristic version:

1. Install the extra dependency:
   ```bash
   pip install mamba-ssm
   ```
2. Ensure checkpoints are present (default expects `ckpt/mosi/best_valid_model_seed_42.pth`).
3. Edit the `user_text` and `user_video` paths at the bottom of `inference.py`.
4. Run:
   ```bash
   python3 inference.py        # macOS / Linux
   ```
   ```powershell
   python inference.py         # Windows
   ```

The full model will report the regression score, label, and which modalities were used.

### 5. Troubleshooting

- **`ModuleNotFoundError: mamba_ssm`** — install `mamba-ssm` or stick to `simplified_inference.py`.
- **`librosa` fails to read `.MOV`** — convert the video to a standard codec or extract audio to `.wav` and adjust the script.
- **`urllib3` OpenSSL warning on macOS** — it is safe to ignore for inference, or install Python compiled against OpenSSL ≥1.1.1.
- **No internet** — manually download `bert-base-uncased` and copy it into the HuggingFace cache before running the script.
