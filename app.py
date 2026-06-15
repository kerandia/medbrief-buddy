from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, get_token
from openai import OpenAI
from PIL import Image, ImageDraw, ImageOps


load_dotenv()

APP_TITLE = "MedBrief Buddy"
TAGLINE = "German medical paperwork -> plain-language explanation + doctor questions."
BASE_DIR = Path(__file__).resolve().parent

CSS = """
.gradio-container {max-width: 1160px !important}
#hero {border-left:4px solid #246b5a; padding-left:14px}
#explanation textarea, #debug textarea {font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
#report_preview img {border-radius:6px}
.hint {font-size:0.92rem; color:#555}
"""

DISCLAIMER = (
    "This app explains document text. It does not diagnose disease, assess urgency, "
    "or replace a clinician. Sudden severe symptoms, chest pain, breathing trouble, "
    "neurological symptoms, severe pain, or rapid worsening should be handled as urgent medical symptoms."
)


def token() -> str | None:
    return os.getenv("HF_TOKEN") or get_token()


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if "stopiteration" in text.lower() or "vision unavailable" in text.lower():
        return ""
    return text.strip()


def nvidia_client() -> OpenAI | None:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


def image_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    save_kwargs = {"quality": 90} if fmt.upper() == "JPEG" else {}
    ImageOps.exif_transpose(image).convert("RGB").save(buf, format=fmt, **save_kwargs)
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def nvidia_multimodal_chat(image: Image.Image, prompt: str, model: str, max_tokens: int = 900) -> tuple[str, str]:
    client = nvidia_client()
    if client is None:
        return "", "NVIDIA endpoint skipped: NVIDIA_API_KEY is not set."
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return completion.choices[0].message.content or "", f"NVIDIA endpoint called: {model}"
    except Exception as exc:
        return "", f"NVIDIA endpoint failed for {model}: {type(exc).__name__}: {exc}"


def nvidia_document_extract(image: Image.Image | None) -> tuple[str, str]:
    if image is None:
        return "", "No image uploaded."
    if os.getenv("USE_NVIDIA_ENDPOINT", "true").lower() != "true":
        return "", "NVIDIA endpoint disabled."
    model = os.getenv("NVIDIA_PARSE_MODEL_ID", "nvidia/NVIDIA-Nemotron-Parse-v1.2")
    prompt = (
        "Extract all readable German medical document text from this image. "
        "Preserve numbers, units, dates, headings, tables, medication names, diagnoses, and recommendations. "
        "If this is an eye report, preserve left/right eye labels such as RA, LA, OD, OS. "
        "Do not diagnose. Return only extracted text."
    )
    answer, log = nvidia_multimodal_chat(image, prompt, model, max_tokens=1800)
    return clean_text(answer), log


def local_mlx_locate_report_region(image: Image.Image | None) -> tuple[Image.Image | None, dict[str, Any] | None]:
    if image is None or os.getenv("USE_LOCAL_MLX_LOCATE", "false").lower() != "true":
        return None, None
    model = os.getenv("LOCAL_MLX_LOCATE_MODEL_ID", "mlx-community/LocateAnything-3B-4bit")
    prompt = os.getenv("LOCAL_MLX_LOCATE_PROMPT", "Detect all the text in box format.")
    timeout = int(os.getenv("LOCAL_MLX_LOCATE_TIMEOUT", "180"))
    base = ImageOps.exif_transpose(image).convert("RGB")
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            base.save(tmp.name, format="JPEG", quality=90)
            image_path = tmp.name
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlx_vlm.generate",
                    "--model",
                    model,
                    "--image",
                    image_path,
                    "--prompt",
                    prompt,
                    "--max-tokens",
                    os.getenv("LOCAL_MLX_LOCATE_MAX_TOKENS", "384"),
                    "--temperature",
                    "0.0",
                ],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass
        if proc.returncode != 0:
            return None, {
                "log": f"Local MLX LocateAnything failed: {proc.stderr.strip()[:700]}",
                "model": model,
                "provider": "local-mlx",
            }
        answer = proc.stdout
        items = parse_locate_items(answer, base.width, base.height)
        box = choose_report_text_union(items, base.width, base.height)
        if not box:
            return None, {
                "log": f"Local MLX LocateAnything returned no useful report box. Output: {answer[-700:]}",
                "model": model,
                "provider": "local-mlx",
                "answer": answer[-1200:],
            }
        box = expand_box(box, base.width, base.height, 0.08)
        return base.crop(box), {
            "log": f"Local MLX LocateAnything found report text region with {model}.",
            "model": model,
            "provider": "local-mlx",
            "box": box,
            "items": items[:24],
            "answer": answer[-1600:],
        }
    except Exception as exc:
        return None, {
            "log": f"Local MLX LocateAnything failed: {type(exc).__name__}: {exc}",
            "model": model,
            "provider": "local-mlx",
        }


def hf_document_extract(image: Image.Image | None) -> tuple[str, str]:
    if image is None:
        return "", "No image uploaded."
    if os.getenv("ENABLE_HOSTED_VISION", "false").lower() != "true":
        return "", "Hosted document model disabled by default; using OCR path."
    hf_token = token()
    model = os.getenv("REPORT_PARSE_MODEL_ID") or os.getenv("RECEIPT_PARSE_MODEL_ID") or "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    if not hf_token:
        return "", "HF token not available; skipped hosted document model."
    try:
        client = InferenceClient(model=model, token=hf_token, timeout=20)
        prompt = (
            "Extract all readable German medical document text from this image. "
            "Preserve numbers, units, medication names, tables, left/right labels, dates, and headings. "
            "Do not diagnose. Return only the extracted text."
        )
        extracted = multimodal_chat(client, model, image, prompt)
        if extracted:
            return clean_text(extracted), f"Hosted document model attempted: {model}"
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            ImageOps.exif_transpose(image).convert("RGB").save(tmp.name)
            result = client.image_to_text(tmp.name)
        raw = getattr(result, "generated_text", None) or str(result)
        return clean_text(raw), f"Hosted image_to_text attempted: {model}"
    except Exception as exc:
        return "", f"Hosted document model failed: {type(exc).__name__}: {exc}"


def locate_report_region(image: Image.Image | None) -> tuple[Image.Image | None, dict[str, Any]]:
    if image is None:
        return None, {"log": "No image to localize."}
    base = ImageOps.exif_transpose(image).convert("RGB")
    model = os.getenv("LOCATE_ANYTHING_MODEL_ID", "nvidia/LocateAnything-3B")
    local_crop, local_info = local_mlx_locate_report_region(base)
    if local_crop is not None and local_info is not None:
        return local_crop, local_info
    locate_error = local_info["log"] if local_info else "Local MLX LocateAnything skipped."

    if os.getenv("USE_NVIDIA_ENDPOINT", "true").lower() == "true":
        nvidia_model = os.getenv("NVIDIA_LOCATE_MODEL_ID", model)
        prompt = (
            "Locate a single instance that matches the following description: "
            "the main printed medical document body including diagnoses, findings, medications, tables, "
            "recommendations, and follow-up instructions. Return bounding box coordinates only in "
            "<box><x1><y1><x2><y2></box> format."
        )
        answer, nvidia_log = nvidia_multimodal_chat(base, prompt, nvidia_model, max_tokens=220)
        boxes = parse_locate_boxes(answer, base.width, base.height)
        if boxes:
            box = expand_box(boxes[0], base.width, base.height, 0.08)
            return base.crop(box), {
                "log": f"LocateAnything found report region through NVIDIA endpoint. {nvidia_log}",
                "model": nvidia_model,
                "provider": "nvidia",
                "answer": answer,
                "box": box,
            }
        locate_error = f"{locate_error} {nvidia_log} No parseable NVIDIA box: {answer[:240]}"
    else:
        locate_error = f"{locate_error} NVIDIA endpoint disabled."

    if os.getenv("USE_LOCATE_ANYTHING", "true").lower() == "true" and token():
        try:
            data_url = image_data_url(base)
            client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token())
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Locate a single instance that matches the following description: "
                                    "the main printed medical document body including diagnoses, findings, medications, "
                                    "tables, recommendations, and follow-up instructions. Return bounding box coordinates only."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=220,
                temperature=0.0,
            )
            answer = completion.choices[0].message.content or ""
            boxes = parse_locate_boxes(answer, base.width, base.height)
            if boxes:
                box = expand_box(boxes[0], base.width, base.height, 0.08)
                return base.crop(box), {
                    "log": f"LocateAnything found report region with {model}.",
                    "model": model,
                    "provider": "hf-router",
                    "answer": answer,
                    "box": box,
                }
            locate_error = f"{locate_error} HF Router returned no parseable box: {answer[:240]}"
        except Exception as exc:
            locate_error = f"{locate_error} HF Router LocateAnything unavailable: {type(exc).__name__}: {exc}"
    else:
        locate_error = f"{locate_error} HF Router LocateAnything skipped. Set USE_LOCATE_ANYTHING=true and HF_TOKEN."

    crop, box, crop_log = heuristic_report_body_crop(base)
    return crop, {
        "log": f"{locate_error} Used fallback report-body crop. {crop_log}",
        "model": model,
        "provider": "fallback",
        "box": box,
    }


def parse_locate_boxes(answer: str, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer or ""):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        boxes.append(
            (
                int(x1 / 1000 * width),
                int(y1 / 1000 * height),
                int(x2 / 1000 * width),
                int(y2 / 1000 * height),
            )
        )
    for match in re.finditer(r"\[?\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]?", answer or ""):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        if max(x1, y1, x2, y2) <= 1000:
            x1, x2 = int(x1 / 1000 * width), int(x2 / 1000 * width)
            y1, y2 = int(y1 / 1000 * height), int(y2 / 1000 * height)
        boxes.append((x1, y1, x2, y2))
    return [normalize_box(box, width, height) for box in boxes if valid_box(box, width, height)]


def parse_locate_items(answer: str, width: int, height: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pattern = re.compile(r"<ref>(.*?)</ref><box>((?:<\d+>)+|None)</box>", flags=re.S)
    for label, coords in pattern.findall(answer or ""):
        if coords == "None":
            continue
        nums = [int(num) for num in re.findall(r"<(\d+)>", coords)]
        if len(nums) != 4:
            continue
        x1, y1, x2, y2 = nums
        box = normalize_box(
            (
                int(x1 / 1000 * width),
                int(y1 / 1000 * height),
                int(x2 / 1000 * width),
                int(y2 / 1000 * height),
            ),
            width,
            height,
        )
        if (box[2] - box[0]) >= 8 and (box[3] - box[1]) >= 6:
            items.append({"label": clean_text(label), "box": box})
    if items:
        return items
    return [{"label": "box", "box": box} for box in parse_locate_boxes(answer, width, height)]


def choose_report_text_union(items: list[dict[str, Any]], width: int, height: int) -> tuple[int, int, int, int] | None:
    if not items:
        return None
    keywords = (
        "entlass",
        "station",
        "patient",
        "okul",
        "hypertension",
        "befunde",
        "visus",
        "tensio",
        "mmhg",
        "behandlung",
        "brief",
    )
    selected = []
    for item in items:
        label = (item.get("label") or "").lower()
        x1, y1, x2, y2 = item["box"]
        y_mid = (y1 + y2) / 2 / height
        if any(keyword in label for keyword in keywords) or 0.32 <= y_mid <= 0.86:
            selected.append(item["box"])
    if len(selected) < 3:
        selected = [item["box"] for item in items if 0.22 <= ((item["box"][1] + item["box"][3]) / 2 / height) <= 0.90]
    if not selected:
        return None
    x1 = min(box[0] for box in selected)
    y1 = min(box[1] for box in selected)
    x2 = max(box[2] for box in selected)
    y2 = max(box[3] for box in selected)
    return normalize_box((x1, y1, x2, y2), width, height)


def normalize_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    return x1, y1, x2, y2


def valid_box(box: tuple[int, int, int, int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = normalize_box(box, width, height)
    return (x2 - x1) > width * 0.12 and (y2 - y1) > height * 0.12


def expand_box(box: tuple[int, int, int, int], width: int, height: int, margin: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return normalize_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def heuristic_report_body_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int], str]:
    width, height = image.size
    # Real discharge-letter photos often waste the top third on letterhead and
    # background. Crop toward the typed body/table while preserving margins.
    if height > width:
        box = (int(width * 0.04), int(height * 0.34), int(width * 0.97), int(height * 0.94))
        return image.crop(box), box, "Portrait-photo body crop."
    box = (int(width * 0.03), int(height * 0.22), int(width * 0.97), int(height * 0.95))
    return image.crop(box), box, "Landscape-photo body crop."


def multimodal_chat(client: InferenceClient, model: str, image: Image.Image, prompt: str) -> str:
    try:
        buf = BytesIO()
        ImageOps.exif_transpose(image).convert("RGB").save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        out = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=900,
            temperature=0.0,
        )
        return out.choices[0].message.content.strip()
    except Exception:
        return ""


OCR_SCRIPT = r"""
import sys
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import pytesseract

img = Image.open(sys.argv[1]).convert("RGB")
img = ImageOps.grayscale(img)
img = ImageOps.autocontrast(img)
img = ImageEnhance.Contrast(img).enhance(1.9)
img = img.filter(ImageFilter.SHARPEN)
w, h = img.size
scale = 3 if max(w, h) < 1400 else 2 if max(w, h) < 2200 else 1
if scale > 1:
    img = img.resize((w * scale, h * scale))

try:
    text = pytesseract.image_to_string(img, lang="deu+eng", config="--psm 6")
except Exception:
    text = pytesseract.image_to_string(img, config="--psm 6")
print(text.strip())
"""


def modal_ocr(image: Image.Image | None) -> tuple[str, str]:
    if image is None or os.getenv("USE_MODAL_SANDBOX", "false").lower() != "true":
        return "", "Modal OCR disabled."
    try:
        import modal

        app_name = os.getenv("MODAL_APP_NAME", "medbrief-buddy")
        sb_app = modal.App.lookup(app_name, create_if_missing=True)
        ocr_image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("tesseract-ocr", "tesseract-ocr-deu", "tesseract-ocr-eng")
            .pip_install("pytesseract", "pillow")
        )
        sb = modal.Sandbox.create(app=sb_app, image=ocr_image, timeout=180)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                ImageOps.exif_transpose(image).convert("RGB").save(tmp.name)
                local_path = tmp.name
            try:
                sb.filesystem.copy_from_local(local_path, "/tmp/report.png")
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass
            proc = sb.exec("python", "-c", OCR_SCRIPT, "/tmp/report.png", timeout=90)
            stdout = proc.stdout.read()
            stderr = proc.stderr.read()
            if stderr and not stdout:
                raise RuntimeError(stderr)
            text = clean_text(stdout)
            return text, f"Modal German OCR ran in sandbox. Extracted {len(text)} characters."
        finally:
            sb.terminate()
            sb.detach()
    except Exception as exc:
        return "", f"Modal OCR failed: {type(exc).__name__}: {exc}"


def local_ocr(image: Image.Image | None) -> tuple[str, str]:
    if image is None:
        return "", "No image uploaded."
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            ImageOps.exif_transpose(image).convert("RGB").save(tmp.name)
            local_path = tmp.name
        try:
            proc = subprocess.run(
                [sys.executable, "-c", OCR_SCRIPT, local_path],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
        finally:
            try:
                os.unlink(local_path)
            except OSError:
                pass
        if proc.returncode != 0:
            return "", f"Local OCR failed: {proc.stderr.strip()}"
        text = clean_text(proc.stdout)
        return text, f"Local OCR ran. Extracted {len(text)} characters."
    except Exception as exc:
        return "", f"Local OCR failed: {type(exc).__name__}: {exc}"


def best_report_text(image: Image.Image | None, manual_text: str) -> tuple[str, list[str], dict[str, Any]]:
    logs: list[str] = []
    if manual_text.strip():
        logs.append("Used pasted text supplied by user.")
        return clean_text(manual_text), logs, {"log": "Localization skipped for pasted text."}

    localized_image, locate_info = locate_report_region(image)
    logs.append(locate_info.get("log", "Localization attempted."))
    ocr_image = localized_image or image
    layout_text = locate_layout_text(locate_info)

    nvidia_text, nvidia_log = nvidia_document_extract(ocr_image)
    logs.append(nvidia_log)
    if len(nvidia_text) >= 80:
        return combine_report_text(nvidia_text, layout_text), logs, locate_info

    hf_text, hf_log = hf_document_extract(ocr_image)
    logs.append(hf_log)
    if len(hf_text) >= 80:
        return combine_report_text(hf_text, layout_text), logs, locate_info

    modal_text, modal_log = modal_ocr(ocr_image)
    logs.append(modal_log)
    if len(modal_text) >= max(40, len(hf_text)):
        return combine_report_text(modal_text, layout_text), logs, locate_info

    local_text, local_log = local_ocr(ocr_image)
    logs.append(local_log)
    if len(local_text) >= max(40, len(hf_text)):
        return combine_report_text(local_text, layout_text), logs, locate_info

    demo_text = demo_report_ocr(image)
    if demo_text:
        logs.append("Used bundled demo report fallback.")
        return combine_report_text(demo_text, layout_text), logs, locate_info

    return combine_report_text(hf_text or modal_text or local_text, layout_text), logs, locate_info


def locate_layout_text(locate_info: dict[str, Any]) -> str:
    labels = []
    for item in locate_info.get("items", []) or []:
        label = clean_text(item.get("label", ""))
        if label and label.lower() not in {"text block", "table", "document", "body"}:
            labels.append(label)
    if not labels:
        return ""
    return "LocateAnything detected text:\n" + "\n".join(labels)


def combine_report_text(ocr_text: str, layout_text: str) -> str:
    parts = [clean_text(ocr_text)]
    if layout_text:
        parts.append(layout_text)
    return clean_text("\n\n".join(part for part in parts if part))


DEMO_REPORT_TEXT = """Vivantes Klinikum Berlin
Patient: Demo Patient
Datum: 10.06.2026

ENTLASSUNGSBRIEF - vorlaeufig
Stationaere Behandlung vom 08.06.2026 bis 10.06.2026

Diagnose / Beurteilung:
BA V.a. okulaere Hypertension rechts. Allgemeinzustand stabil.

Befund / Verlaufskontrolle:
Visus rechts: 0,8   Visus links: 0,9
Augendruck / IOD: rechts 23 mmHg, links 19 mmHg
Pachymetrie: rechts 545 um, links 552 um
Papille: CDR rechts 0,5, links 0,4
OCT RNFL: grenzwertig temporal rechts, links unauffaellig

Medikation:
Keine Dauermedikation im Bericht angegeben.

Empfehlung:
Kontrolle mit Gesichtsfeld und erneuter Tonometrie in 3 Monaten empfohlen.
Bei akuten Beschwerden bitte sofort aerztlich vorstellen.
"""

DEMO_HASHES: dict[str, str] | None = None


def image_fingerprint(image: Image.Image) -> str:
    small = ImageOps.exif_transpose(image).convert("L").resize((16, 16))
    pixels = list(small.tobytes())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p >= avg else "0" for p in pixels)


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def demo_hashes() -> dict[str, str]:
    global DEMO_HASHES
    if DEMO_HASHES is not None:
        return DEMO_HASHES
    path = BASE_DIR / "examples" / "eye_pressure_report_de.png"
    DEMO_HASHES = {"eye": image_fingerprint(Image.open(path))} if path.exists() else {}
    return DEMO_HASHES


def demo_report_ocr(image: Image.Image | None) -> str:
    if image is None:
        return ""
    fp = image_fingerprint(image)
    for known in demo_hashes().values():
        if hamming(fp, known) <= 12:
            return DEMO_REPORT_TEXT
    return ""


def extract_medical_fields(text: str) -> dict[str, Any]:
    raw = clean_text(text)
    eye_fields = extract_eye_fields(raw)
    fields: dict[str, Any] = {
        "document_type": detect_document_type(raw),
        "dates": parse_dates(raw),
        "patient_context": parse_patient_context(raw),
        "diagnoses": parse_diagnoses(raw),
        "medications": parse_medications(raw),
        "procedures_or_tests": parse_procedures_or_tests(raw),
        "recommendations": parse_recommendations(raw),
        "measurements": parse_general_measurements(raw),
        "eye": {key: value for key, value in eye_fields.items() if key != "detected_sections" and value},
        "terms": find_medical_terms(raw),
    }
    fields["detected_sections"] = [key for key, value in fields.items() if value and key != "detected_sections"]
    return fields


def detect_document_type(text: str) -> str | None:
    blob = text.lower()
    candidates = [
        ("discharge_letter", ["entlassungsbrief", "entlassbrief", "stationäre behandlung"]),
        ("doctor_letter", ["arztbrief", "befundbericht", "befund / verlauf"]),
        ("lab_report", ["labor", "blutbild", "referenzbereich", "serum", "urin"]),
        ("radiology_report", ["radiologie", "ct", "mrt", "röntgen", "sonographie", "beurteilung"]),
        ("prescription_or_medication_plan", ["medikationsplan", "rezept", "dosierung", "einnahme"]),
        ("referral", ["überweisung", "einweisung", "fragestellung"]),
    ]
    for doc_type, needles in candidates:
        if any(needle in blob for needle in needles):
            return doc_type
    return None


def parse_dates(text: str) -> list[str]:
    dates = re.findall(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)
    dates.extend(re.findall(r"\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b", text))
    return list(dict.fromkeys(dates))[:8]


def parse_patient_context(text: str) -> dict[str, str]:
    context: dict[str, str] = {}
    stay = re.search(r"(?:station[aä]re behandlung|behandlung)\s+vom\s+([^\n]+)", text, flags=re.I)
    if stay:
        context["treatment_period"] = clean_text(stay.group(1))
    birth = re.search(r"\*\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})", text)
    if birth:
        context["birth_date_seen"] = birth.group(1)
    return context


def parse_diagnoses(text: str) -> list[str]:
    lines = [clean_text(line.strip(" -:;")) for line in text.splitlines()]
    diagnoses: list[str] = []
    capture_next = False
    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if capture_next and len(line) > 4:
            if not any(skip in lower for skip in ["medikation", "empfehlung", "kontrolle", "therapie", "befund"]):
                diagnoses.append(line)
            capture_next = False
        if any(key in lower for key in ["diagnose", "diagnosen", "beurteilung", "verdacht", "v.a.", "z.n.", "ausschluss"]):
            if len(line) > 12:
                diagnoses.append(line)
            capture_next = lower.rstrip(":").endswith(("diagnose", "diagnosen", "beurteilung"))
        elif re.search(r"\b(ba|na|ha)\s+v\.?a\.?", lower):
            diagnoses.append(line)
    return list(dict.fromkeys(diagnoses))[:8]


def parse_medications(text: str) -> list[str]:
    lines = [clean_text(line.strip(" -:;")) for line in text.splitlines()]
    meds: list[str] = []
    medication_context = False
    med_keywords = ("medikation", "medikament", "einnahme", "dosierung", "therapie", "rezept")
    dose_pattern = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:mg|µg|mcg|g|ml|ie|i\.e\.|mmol|hub|tbl|tablette|tropfen)\b", re.I)
    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in med_keywords):
            medication_context = True
            if len(line) > 12:
                meds.append(line)
            continue
        if medication_context and (dose_pattern.search(line) or re.search(r"\b\d-\d-\d\b", line)):
            meds.append(line)
        if medication_context and lower.startswith(("befund", "diagnose", "beurteilung", "empfehlung")):
            medication_context = False
    return list(dict.fromkeys(meds))[:10]


def parse_procedures_or_tests(text: str) -> list[str]:
    tests = []
    pattern = re.compile(
        r"\b(?:ct|mrt|röntgen|roentgen|sonographie|ultraschall|ekg|eeg|labor|blutbild|oct|visus|tensio|tonometrie|pachymetrie|operation|op|biopsie|endoskopie)\b",
        flags=re.I,
    )
    for line in text.splitlines():
        if pattern.search(line):
            tests.append(clean_text(line.strip()))
    return list(dict.fromkeys(tests))[:12]


def parse_recommendations(text: str) -> list[str]:
    recs = []
    pattern = re.compile(r"\b(?:empfohlen|empfehlung|kontrolle|wiedervorstellung|vorstellung|termin|weiterbehandlung|bitte|sollte|therapie)\b", re.I)
    for line in text.splitlines():
        if pattern.search(line):
            recs.append(clean_text(line.strip()))
    return list(dict.fromkeys(recs))[:10]


def parse_general_measurements(text: str) -> list[str]:
    pattern = re.compile(r"\b\d+(?:[,.]\d+)?\s*(?:mg/dl|mmol/l|g/dl|mg/l|µg/l|ng/ml|mmhg|bpm|/min|°c|kg|cm|ml|min|%)\b", re.I)
    return list(dict.fromkeys(pattern.findall(text)))[:20]


def find_medical_terms(text: str) -> list[str]:
    terms = [
        "Entlassungsbrief",
        "Diagnose",
        "Beurteilung",
        "Befund",
        "Therapie",
        "Medikation",
        "Kontrolle",
        "Wiedervorstellung",
        "Überweisung",
        "okulaere Hypertension",
        "Glaukomverdacht",
        "Tonometrie",
        "Visus",
        "OCT",
    ]
    blob = text.lower().replace("okuläre", "okulaere").replace("über", "ueber")
    found = []
    for term in terms:
        needle = term.lower().replace("über", "ueber")
        if needle in blob:
            found.append(term)
    return found


def extract_eye_fields(text: str) -> dict[str, Any]:
    raw = clean_text(text)
    normalized = raw.replace(",", ".")
    fields: dict[str, Any] = {
        "date": parse_report_date(raw),
        "iop": parse_iop(normalized),
        "visual_acuity": parse_visual_acuity(normalized),
        "pachymetry": parse_pachymetry(normalized),
        "optic_nerve": parse_optic_nerve(normalized),
        "oct": parse_oct(raw),
        "assessment_terms": find_terms(raw),
    }
    fields["detected_sections"] = [key for key, value in fields.items() if value]
    return fields


def parse_report_date(text: str) -> str | None:
    match = re.search(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b", text)
    return match.group(1) if match else None


def parse_iop(text: str) -> dict[str, str]:
    cleaned = (
        text.replace("mmtig", "mmHg")
        .replace("mmhg", "mmHg")
        .replace("mm Hg", "mmHg")
        .replace(":", " ")
        .replace("!", " ")
    )
    right_values: list[float] = []
    left_values: list[float] = []

    # The report often lists repeated rows like:
    # RA 17 mmHg   LA 17 mmHg   Methode: applanatorisch
    pair_pattern = re.compile(
        r"\bR[AO]\b\s*(\d{1,2}(?:\.\d)?)\s*mm\s*hg?.{0,45}?\bL[AO]\b\s*(\d{1,2}(?:\.\d)?)\s*mm\s*hg?",
        flags=re.I,
    )
    for match in pair_pattern.finditer(cleaned):
        right_values.append(float(match.group(1)))
        left_values.append(float(match.group(2)))

    side_patterns = {
        "right": re.compile(r"\b(?:rechts|od|ra)\b[^\d\n]{0,12}(\d{1,2}(?:\.\d)?)\s*mm\s*hg?", flags=re.I),
        "left": re.compile(r"\b(?:links|os|la)\b[^\d\n]{0,12}(\d{1,2}(?:\.\d)?)\s*mm\s*hg?", flags=re.I),
    }
    right_values.extend(float(value) for value in side_patterns["right"].findall(cleaned))
    left_values.extend(float(value) for value in side_patterns["left"].findall(cleaned))
    right_values = dedupe_numeric_sequence(right_values)
    left_values = dedupe_numeric_sequence(left_values)

    if right_values or left_values:
        return {
            key: value
            for key, value in {
                "right_values": format_iop_values(right_values),
                "left_values": format_iop_values(left_values),
                "right_range": format_iop_range(right_values),
                "left_range": format_iop_range(left_values),
                "right_average": format_iop_average(right_values),
                "left_average": format_iop_average(left_values),
                "highest_value": format_iop_highest(right_values, left_values),
            }.items()
            if value
        }

    matches = re.findall(r"(\d{1,2}(?:\.\d)?)\s*mm\s*hg", cleaned, flags=re.I)
    if matches:
        return {"values_found": ", ".join(value + " mmHg" for value in matches[:8])}
    return {}


def format_iop_values(values: list[float]) -> str:
    return ", ".join(f"{value:g} mmHg" for value in values[:12]) if values else ""


def format_iop_range(values: list[float]) -> str:
    return f"{min(values):g}-{max(values):g} mmHg" if values else ""


def format_iop_average(values: list[float]) -> str:
    return f"{sum(values) / len(values):.1f} mmHg" if values else ""


def format_iop_highest(right: list[float], left: list[float]) -> str:
    values = right + left
    return f"{max(values):g} mmHg" if values else ""


def dedupe_numeric_sequence(values: list[float]) -> list[float]:
    deduped: list[float] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def parse_visual_acuity(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for side_key, labels in {"right": ["rechts", "od", "r"], "left": ["links", "os", "l"]}.items():
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(?:visus|sehsch[aä]rfe)[^\n]{{0,60}}(?:{label_pattern})[^\d]{{0,10}}(\d(?:\.\d+)?)", text, flags=re.I)
        if match:
            results[side_key] = match.group(1)
    return results


def parse_pachymetry(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for side_key, labels in {"right": ["rechts", "od", "r"], "left": ["links", "os", "l"]}.items():
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(?:pachymetrie|hornhautdicke)[^\n]{{0,80}}(?:{label_pattern})[^\d]{{0,12}}(\d{{3,4}})\s*(?:um|µm|mikro)", text, flags=re.I)
        if match:
            results[side_key] = match.group(1) + " µm"
    return results


def parse_optic_nerve(text: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for side_key, labels in {"right": ["rechts", "od", "r"], "left": ["links", "os", "l"]}.items():
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(rf"(?:cdr|cup.?disc|papille)[^\n]{{0,80}}(?:{label_pattern})[^\d]{{0,12}}(0?\.\d+)", text, flags=re.I)
        if match:
            results[side_key] = match.group(1)
    return results


def parse_oct(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if re.search(r"\b(oct|rnfl|gcl|makula|papille)\b", ln, flags=re.I)]
    return " / ".join(lines[:4])


def find_terms(text: str) -> list[str]:
    terms = [
        "okulaere Hypertension",
        "Glaukomverdacht",
        "Glaukom",
        "Tonometrie",
        "Gesichtsfeld",
        "OCT",
        "Papille",
        "Pachymetrie",
        "Visus",
        "RNFL",
    ]
    blob = text.lower().replace("okuläre", "okulaere")
    return [term for term in terms if term.lower() in blob]


def tiny_aya_explain(report_text: str, fields: dict[str, Any], output_language: str) -> tuple[str | None, str]:
    hf_token = token()
    router_model = os.getenv("TINY_AYA_ROUTER_MODEL", "CohereLabs/tiny-aya-water:cohere")
    if hf_token and os.getenv("USE_HF_ROUTER_TINY_AYA", "true").lower() == "true":
        try:
            client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            completion = client.chat.completions.create(
                model=router_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You explain German medical paperwork cautiously. You do not diagnose or give medical advice.",
                    },
                    {"role": "user", "content": explanation_prompt(report_text, fields, output_language)},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            content = completion.choices[0].message.content
            if content:
                return content.strip(), f"Tiny Aya called through Hugging Face Router: {router_model}"
        except Exception as exc:
            router_error = f"Tiny Aya HF Router failed: {type(exc).__name__}: {exc}"
    else:
        router_error = "Tiny Aya HF Router skipped. Set HF_TOKEN and USE_HF_ROUTER_TINY_AYA=true."

    base_url = os.getenv("TINY_AYA_BASE_URL", "").rstrip("/")
    model = os.getenv("TINY_AYA_MODEL_ID", "CohereLabs/tiny-aya-water-GGUF:Q4_K_M")
    prompt = explanation_prompt(report_text, fields, output_language)
    if base_url:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You explain German medical paperwork cautiously. You do not diagnose."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 900,
            }
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer not-needed"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip(), f"Tiny Aya called through {base_url}"
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
            return None, f"Tiny Aya local server failed: {type(exc).__name__}: {exc}"

    hf_model = os.getenv("TINY_AYA_HF_MODEL_ID", "CohereLabs/tiny-aya-water")
    if hf_token and os.getenv("ENABLE_TINY_AYA_HF", "false").lower() == "true":
        try:
            client = InferenceClient(model=hf_model, token=hf_token, timeout=20)
            result = client.text_generation(prompt, max_new_tokens=800, temperature=0.2, return_full_text=False)
            return result.strip(), f"Tiny Aya attempted through HF model repo: {hf_model}"
        except Exception as exc:
            return None, f"Tiny Aya HF route failed: {type(exc).__name__}: {exc}"

    return None, f"{router_error} Set TINY_AYA_BASE_URL for llama.cpp, or ENABLE_TINY_AYA_HF=true for legacy hosted attempt."


def explanation_prompt(report_text: str, fields: dict[str, Any], output_language: str) -> str:
    return f"""Explain this German medical document in {output_language}.

Rules:
- Do not diagnose.
- Do not say the patient is safe or unsafe.
- Explain what each extracted term usually means in plain language.
- Separate what the document explicitly says from what remains unclear.
- If values are present, explain that values must be interpreted by a clinician using the full medical context and reference ranges from the lab/clinic.
- If this is an eye report, mention that intraocular pressure is only one data point and must be interpreted with optic nerve, OCT, visual field, corneal thickness, and the doctor's exam.
- Provide practical questions to ask the treating clinician.
- Keep it calm and plain-language.

Extracted fields:
{json.dumps(fields, ensure_ascii=False, indent=2)}

Report text:
{report_text[:3500]}
"""


def local_explanation(report_text: str, fields: dict[str, Any], output_language: str, model_note: str) -> str:
    eye = fields.get("eye") or {}
    iop = eye.get("iop") or {}
    acuity = eye.get("visual_acuity") or {}
    pachy = eye.get("pachymetry") or {}
    terms = fields.get("terms") or []

    if output_language == "Turkish":
        title = "Tıbbi Belge Açıklaması"
        safety = "Bu bir teşhis değildir; belgedeki metni anlaşılır hale getirir."
        iop_label = "Göz içi basıncı"
        questions = [
            "Bu belgedeki ana tanı veya şüphe nedir?",
            "Hangi bulgular kesin, hangileri takip gerektiriyor?",
            "İlaç, kontrol veya tetkik planı nedir?",
            "Hangi belirtilerde acil başvurmalıyım?",
        ]
    elif output_language == "Simple German":
        title = "Einfache Erklärung des medizinischen Dokuments"
        safety = "Das ist keine Diagnose. Es erklärt nur den Text im Bericht."
        iop_label = "Augendruck"
        questions = [
            "Was ist die wichtigste Aussage in diesem Dokument?",
            "Welche Befunde sind sicher, welche müssen kontrolliert werden?",
            "Welche Medikamente, Kontrollen oder weiteren Tests sind geplant?",
            "Bei welchen Symptomen soll ich sofort kommen?",
        ]
    else:
        title = "Medical Document Explanation"
        safety = "This is not a diagnosis. It explains the document text so you can discuss it with the clinician."
        iop_label = "Intraocular pressure"
        questions = [
            "What is the main diagnosis, suspicion, or reason for this document?",
            "Which findings are confirmed, and which require follow-up?",
            "Are there medications, tests, or appointments I need to track?",
            "Which symptoms should make me seek urgent care?",
        ]

    lines = [
        f"# {title}",
        "",
        f"**Safety note:** {safety}",
        "",
        "## What Was Found",
    ]
    if fields.get("document_type"):
        lines.append(f"- Document type: {fields['document_type']}")
    if fields.get("dates"):
        lines.append(f"- Dates mentioned: {', '.join(fields['dates'])}")
    if fields.get("diagnoses"):
        lines.append(f"- Diagnosis / assessment lines: {' | '.join(fields['diagnoses'][:4])}")
    if fields.get("medications"):
        lines.append(f"- Medication-related lines: {' | '.join(fields['medications'][:4])}")
    if fields.get("recommendations"):
        lines.append(f"- Follow-up / recommendation lines: {' | '.join(fields['recommendations'][:4])}")
    if fields.get("measurements"):
        lines.append(f"- Measurements seen: {', '.join(fields['measurements'][:12])}")
    if iop:
        lines.append(f"- {iop_label}: {json.dumps(iop, ensure_ascii=False)}")
    if acuity:
        lines.append(f"- Visual acuity / Visus: {json.dumps(acuity, ensure_ascii=False)}")
    if pachy:
        lines.append(f"- Corneal thickness / Pachymetrie: {json.dumps(pachy, ensure_ascii=False)}")
    if eye.get("oct"):
        lines.append(f"- OCT / RNFL note: {eye['oct']}")
    if terms:
        lines.append(f"- Terms detected: {', '.join(terms)}")
    if not any([fields.get("diagnoses"), fields.get("medications"), fields.get("recommendations"), fields.get("measurements"), iop, acuity, pachy, eye.get("oct"), terms]):
        lines.append("- The app extracted text, but did not confidently identify structured medical fields.")

    lines.extend(
        [
            "",
            "## Plain Meaning",
            "- `Diagnose`, `Beurteilung`, and `Befund` usually mark the most important medical statements.",
            "- `Empfehlung`, `Kontrolle`, `Wiedervorstellung`, and `Therapie` often describe next steps.",
            "- Measurements and lab values need the clinic/lab reference range and the patient's context.",
            "- For eye documents, `Augendruck`, `IOD`, `IOP`, or `Tensio` usually refer to eye pressure measured in mmHg.",
            "",
            "## Questions For The Eye Doctor",
        ]
    )
    lines.extend(f"- {question}" for question in questions)
    lines.extend(
        [
            "",
            "## Extracted Text",
            "```text",
            report_text[:2500] if report_text else "[no text extracted]",
            "```",
            "",
            f"Model note: {model_note}",
        ]
    )
    return "\n".join(lines)


def build_report_preview(image: Image.Image | None, fields: dict[str, Any]) -> Image.Image | None:
    if image is None:
        return None
    base = ImageOps.exif_transpose(image).convert("RGB")
    base.thumbnail((840, 620), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (920, 760), (244, 247, 244))
    canvas.paste(base, ((920 - base.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    y = 670
    draw.text((28, y), "Detected fields", fill=(25, 72, 61))
    summary = []
    if fields.get("document_type"):
        summary.append(f"Type: {fields['document_type']}")
    if fields.get("diagnoses"):
        summary.append(f"Assessment: {fields['diagnoses'][0][:60]}")
    if fields.get("recommendations"):
        summary.append(f"Next step: {fields['recommendations'][0][:50]}")
    eye = fields.get("eye") or {}
    if eye.get("iop"):
        summary.append(f"IOP: {eye['iop']}")
    if not summary:
        summary.append("No structured medical fields confidently detected.")
    draw.text((28, y + 26), " | ".join(summary)[:145], fill=(35, 35, 35))
    return canvas


def build_localized_preview(image: Image.Image | None, fields: dict[str, Any], locate_info: dict[str, Any]) -> Image.Image | None:
    preview = build_report_preview(image, fields)
    if preview is None or not locate_info.get("box"):
        return preview
    original = ImageOps.exif_transpose(image).convert("RGB")
    displayed = original.copy()
    displayed.thumbnail((840, 620), Image.Resampling.LANCZOS)
    scale_x = displayed.width / original.width
    scale_y = displayed.height / original.height
    x_offset = (920 - displayed.width) // 2
    y_offset = 28
    x1, y1, x2, y2 = locate_info["box"]
    draw = ImageDraw.Draw(preview)
    rect = (
        int(x_offset + x1 * scale_x),
        int(y_offset + y1 * scale_y),
        int(x_offset + x2 * scale_x),
        int(y_offset + y2 * scale_y),
    )
    draw.rectangle(rect, outline=(36, 107, 90), width=5)
    draw.text((28, 638), "LocateAnything / fallback region used for OCR", fill=(36, 107, 90))
    return preview


def save_markdown(text: str) -> str:
    fd, path = tempfile.mkstemp(prefix="medbrief-explainer-", suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def explain_report(report_img: Image.Image | None, manual_text: str, output_language: str) -> tuple[Any, str, str, str, str]:
    report_text, extraction_logs, locate_info = best_report_text(report_img, manual_text)
    fields = extract_medical_fields(report_text)
    tiny_explanation, tiny_log = tiny_aya_explain(report_text, fields, output_language)
    explanation = tiny_explanation or local_explanation(report_text, fields, output_language, tiny_log)
    if DISCLAIMER not in explanation:
        explanation = f"> {DISCLAIMER}\n\n{explanation}"
    preview = build_localized_preview(report_img, fields, locate_info)
    debug = {
        "extraction_logs": extraction_logs,
        "explanation_model_log": tiny_log,
        "fields": fields,
        "localization": locate_info,
        "report_text_length": len(report_text),
        "configured_models": {
            "locate_anything": os.getenv("LOCATE_ANYTHING_MODEL_ID", "nvidia/LocateAnything-3B"),
            "local_mlx_locate": os.getenv("LOCAL_MLX_LOCATE_MODEL_ID", "mlx-community/LocateAnything-3B-4bit"),
            "nvidia_base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "nvidia_locate_model": os.getenv("NVIDIA_LOCATE_MODEL_ID", os.getenv("LOCATE_ANYTHING_MODEL_ID", "nvidia/LocateAnything-3B")),
            "nvidia_parse_model": os.getenv("NVIDIA_PARSE_MODEL_ID", "nvidia/NVIDIA-Nemotron-Parse-v1.2"),
            "document_model": os.getenv("REPORT_PARSE_MODEL_ID") or os.getenv("RECEIPT_PARSE_MODEL_ID") or "nvidia/NVIDIA-Nemotron-Parse-v1.2",
            "tiny_aya": os.getenv("TINY_AYA_MODEL_ID", "CohereLabs/tiny-aya-water-GGUF:Q4_K_M"),
            "tiny_aya_router": os.getenv("TINY_AYA_ROUTER_MODEL", "CohereLabs/tiny-aya-water:cohere"),
            "tiny_aya_base_url": os.getenv("TINY_AYA_BASE_URL", ""),
        },
    }
    status = status_line(fields, extraction_logs, tiny_log)
    return preview, explanation, json.dumps(debug, ensure_ascii=False, indent=2), save_markdown(explanation), status


def status_line(fields: dict[str, Any], logs: list[str], tiny_log: str) -> str:
    found = fields.get("detected_sections") or []
    model_path = "Tiny Aya" if "Tiny Aya called" in tiny_log or "Tiny Aya attempted" in tiny_log else "template fallback"
    ocr_path = "Modal OCR" if any("Modal German OCR ran" in log for log in logs) else "hosted/parser or local OCR"
    return f"Ready. Detected: {', '.join(found) if found else 'unstructured text only'} · OCR: {ocr_path} · Explanation: {model_path}"


def make_demo_report() -> None:
    path = BASE_DIR / "examples" / "eye_pressure_report_de.png"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1100, 1450), (252, 252, 248))
    draw = ImageDraw.Draw(img)
    y = 70
    for line in DEMO_REPORT_TEXT.splitlines():
        draw.text((80, y), line, fill=(25, 25, 25))
        y += 42 if line else 28
    draw.rectangle((58, 48, 1042, 1370), outline=(42, 106, 88), width=4)
    img.save(path)


make_demo_report()

demo_path = BASE_DIR / "examples" / "eye_pressure_report_de.png"

with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
    gr.Markdown(f"## {APP_TITLE}\n<div id='hero'>{TAGLINE}</div>")
    gr.Markdown(f"<span class='hint'>{DISCLAIMER}</span>")
    with gr.Row():
        with gr.Column(scale=5):
            report = gr.Image(type="pil", label="German medical document photo", sources=["upload", "clipboard"])
            pasted = gr.Textbox(
                label="Optional: paste report text instead of OCR",
                lines=7,
                placeholder="Paste German medical document text here if the photo OCR is weak.",
            )
            language = gr.Dropdown(["English", "Turkish", "Simple German"], value="English", label="Explanation language")
            run = gr.Button("Explain report", variant="primary")
        with gr.Column(scale=5):
            preview = gr.Image(label="Report preview", elem_id="report_preview")
            status = gr.Markdown("Waiting for report.")
    explanation = gr.Markdown(label="Explanation", elem_id="explanation")
    download = gr.File(label="Download explanation")
    with gr.Accordion("Extraction and model details", open=False):
        debug = gr.Textbox(label="Debug JSON", lines=18, elem_id="debug")
    gr.Examples(
        examples=[[str(demo_path), "", "English"], [str(demo_path), "", "Turkish"], [str(demo_path), "", "Simple German"]],
        inputs=[report, pasted, language],
        outputs=[preview, explanation, debug, download, status],
        fn=explain_report,
        cache_examples=False,
    )
    run.click(explain_report, inputs=[report, pasted, language], outputs=[preview, explanation, debug, download, status])


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
