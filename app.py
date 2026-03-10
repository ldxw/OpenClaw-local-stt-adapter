import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Local STT Adapter")

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "zh")
API_TOKEN = os.getenv("API_TOKEN", "").strip()

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
async def healthz(authorization: Optional[str] = Header(default=None)):
    check_auth(authorization)
    return {
        "ok": True,
        "model": MODEL_SIZE,
        "device": DEVICE,
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(None),
    authorization: Optional[str] = Header(default=None),
):
    check_auth(authorization)

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    suffix = os.path.splitext(file.filename or "audio.ogg")[1] or ".ogg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language or DEFAULT_LANGUAGE,
            initial_prompt=prompt,
            vad_filter=True,
            beam_size=5,
        )
        text = "".join(seg.text for seg in segments).strip()

        if response_format == "text":
            return JSONResponse(content={"text": text})

        return JSONResponse(
            content={
                "text": text,
                "language": info.language,
                "duration": info.duration,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
