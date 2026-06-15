---
title: MedBrief Buddy
emoji: 🏥
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# MedBrief Buddy

Live Space: https://huggingface.co/spaces/build-small-hackathon/medbrief-buddy

Upload a German medical document photo: discharge letter, doctor letter, lab
report, referral, medication plan, or a specialty report such as an eye report.
The app extracts the text, identifies structured medical sections, and explains
the document in plain English, Turkish, or simple German.

The app is intentionally not a diagnosis system. It helps a patient or caregiver
understand what the document says, what is unclear, and what to ask the treating
clinician.

## Track Strategy

Primary: **Backyard AI**. The specific user is someone who receives German
medical paperwork and cannot confidently understand the medical language,
follow-up instructions, dates, measurements, or diagnosis wording.

Secondary: **Best Use of Modal** and **Cohere Models**. Modal Sandbox provides
German OCR fallback. Tiny Aya Water is the small multilingual explanation model.

## Small Model Stack

- `nvidia/LocateAnything-3B` for document-region and text-layout grounding.
- `mlx-community/LocateAnything-3B-4bit` locally on Apple Silicon through
  `mlx-vlm` for the same localization step.
- `nvidia/NVIDIA-Nemotron-Parse-v1.2` as the document/vision parsing slot when
  available.
- Modal Sandbox + German Tesseract OCR as a reliable fallback.
- `CohereLabs/tiny-aya-water:cohere` through the Hugging Face Router for German
  explanation and translation.
- `CohereLabs/tiny-aya-water-GGUF` through a local `llama.cpp` server as an
  optional fallback.

All models are below the 32B hackathon limit.

## Run

```bash
cp .env.example .env
# add HF_TOKEN and Modal env vars/secrets as needed
pip install -r requirements.txt
python app.py
```

The submitted Hugging Face Space runs on CPU with:

```text
Tesseract German OCR -> structured medical extraction -> Tiny Aya via HF Router
```

The local Mac demo can additionally enable:

```text
Local MLX LocateAnything -> OCR -> Tiny Aya via HF Router
```

Local isolated run from this temp folder:

```bash
cd /private/tmp/damage-claim-agent
/private/tmp/damage-claim-agent/.venv/bin/python app.py
```

Enable Modal German OCR:

```bash
USE_MODAL_SANDBOX=true
MODAL_APP_NAME=medbrief-buddy
```

Preferred hosted Tiny Aya route:

```bash
USE_HF_ROUTER_TINY_AYA=true
TINY_AYA_ROUTER_MODEL=CohereLabs/tiny-aya-water:cohere
```

Optional local LocateAnything on Apple Silicon:

```bash
pip install "git+https://github.com/beshkenadze/mlx-vlm@feat/locateanything-3b"
USE_LOCAL_MLX_LOCATE=true
LOCAL_MLX_LOCATE_MODEL_ID=mlx-community/LocateAnything-3B-4bit
```

For Hugging Face Spaces, add secrets:

- `HF_TOKEN`
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `USE_MODAL_SANDBOX=true`
- `MODAL_APP_NAME=medbrief-buddy`
- `USE_HF_ROUTER_TINY_AYA=true`
- `TINY_AYA_ROUTER_MODEL=CohereLabs/tiny-aya-water:cohere`
- `USE_LOCAL_MLX_LOCATE=false` on Linux Spaces unless you have an Apple Silicon runtime
- `USE_LOCATE_ANYTHING=true`
- `LOCATE_ANYTHING_MODEL_ID=nvidia/LocateAnything-3B`
- `NVIDIA_API_KEY` only if a matching NVIDIA endpoint is available
- `TINY_AYA_BASE_URL` only if you run Tiny Aya behind an accessible endpoint

LocateAnything's NVIDIA license is research/non-commercial. Keep that limitation
visible in the demo and submission notes.
