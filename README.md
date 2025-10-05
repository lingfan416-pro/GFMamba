# 🎭 GFMamba: Multimodal Sentiment Analysis with Mamba-based Fusion (CS705 Group 12 Revised Version)

This project is a modified version based on the content of **https://github.com/zzhe232/GFMamba.git**. It includes the following contents
- A PyTorch-based implementation of **multimodal sentiment analysis**, using the **GFMamba** model that integrates **text**, **audio**, and **vision** features for tasks like CMU-MOSI and CMU-MOSEI.
- A **graphical desktop program** is made according to the project plan of CS705 Group 12, which is convenient for data collection during the experiment and batch processing of files.
- Two **automated processing scripts** were developed because the complete model was developed solely based on Linux + NVIDIA CUDA. To enable the model to be used on general devices, we created a simplified inference version of the automated processing script and a full modle inference version (linux/MacOS) of the automated processing script.

if you interested in GFMamba project ,concat us by email zzhe232@aucklanduni.ac.nz

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

```{bash}
git clone https://github.com/yourusername/gfmamba.git
cd gfmamba
pip install -r requirements.txt
```
## requirement

```{bash}
pip install pytorch>2.1 python>3.9
```
---

## 🚀 Quick Start (macOS & Windows)

The repository includes a simplified inference script that lets you run multimodal sentiment analysis on your own `.MOV` video and `.txt` transcript without any extra training. Share these instructions with teammates so they can reproduce the same workflow on macOS or Windows.

### 1. Prepare the environment

| Platform | Command |
| --- | --- |
| **macOS / Linux (Bash)** | `python3 -m venv .venv && source .venv/bin/activate` |
| **Windows (PowerShell)** | `python -m venv .venv; .\.venv\Scripts\Activate.ps1`

Then install dependencies inside the virtual environment:

```{bash}
pip install -r requirements.txt
```

> ℹ️ **Note:** The first run downloads the `bert-base-uncased` checkpoint from HuggingFace. Ensure the machine has internet access or pre-populate the cache (`~/.cache/huggingface` on macOS/Linux, `%USERPROFILE%\.cache\huggingface` on Windows).

### 2. Preparation of the input files

Only the video file(.MOV/.MP4) needs to be provided. The project has an automatic script that can obtain the audio file(.WAV) and interview transcript(.TXT) of the video.

### 3. Run the simplified inference

From the project root (`GFMamba-main`):

```{bash}
python3 simplified_main.py   # macOS / Linux
```

```{powershell}
python simplified_main.py    # Windows
```
This is a graphical desktop program.
- Allows users to select local video files (such as MP4, MOV) through a visual interface.
- Automatically performs a series of processes on the selected videos, including archiving, transcription, and sentiment analysis.
- The archived and analyzed files will be automatically saved to the designated folder, making it convenient for users to access and manage them.

The script prints:

- Device information (CPU/GPU)
- Extracted feature shapes for text, audio, and video
- Sentiment scores for each modality and the fused output
- The final label (Positive / Neutral / Negative)

It also writes the latest result to `simplified_result.json` in the repository root. All the results will be automatically downloaded to the default download path of the computer and saved in the archived_video folder according to the timestamp.

### 4. Optional: Use the full GFMamba model(Linux/MacOS only)

In the CPU-only environment of Windows (torch==2.8.0+cpu), when running "pip install mamba-ssm", pip failed to find a wheel that is compatible with the platform. Therefore, it switched to compiling from source code, which made it difficult to use mamba-ssm on Windows.
If you want to use the full model instead of the heuristic version, you need to have the following configurations: # Linux + NVIDIA GPU + CUDA or # MacOS:

1. Install the extra dependency:
   ```{bash}
   pip install mamba-ssm
   ```
2. Ensure checkpoints are present (default expects `ckpt/mosi/best_valid_model_seed_42.pth`).
3. Edit the `user_text` and `user_video` paths at the bottom of `inference.py`.
4. Run:
   ```{bash}
   python3 inference.py        # macOS / Linux
   ```
   ```{powershell}
   python inference.py         # Windows
   ```

The full model will report the regression score, label, and which modalities were used.

### 5. Troubleshooting

- **`ModuleNotFoundError: mamba_ssm`** — install `mamba-ssm` or stick to `simplified_inference.py`.
- **`librosa` fails to read `.MOV`** — convert the video to a standard codec or extract audio to `.wav` and adjust the script.
- **`urllib3` OpenSSL warning on macOS** — it is safe to ignore for inference, or install Python compiled against OpenSSL ≥1.1.1.
- **No internet** — manually download `bert-base-uncased` and copy it into the HuggingFace cache before running the script.


## License
This project is based on [OriginalProjectName]([https://github.com/user/originalrepo](https://github.com/zzhe232/GFMamba)),  
licensed under the MIT License.  
Modifications © 2025 YourName.
