import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel

app = FastAPI(title="Local STT Adapter")

MODEL_SIZE = os.getenv("WHISPER_MODEL") or "small"
DEVICE = os.getenv("WHISPER_DEVICE") or "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or "int8"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or "zh"
API_TOKEN = (os.getenv("API_TOKEN") or "").strip()

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/", response_class=HTMLResponse)
async def index():
    token_hint = "已启用 Bearer Token 鉴权" if API_TOKEN else "未启用鉴权"
    html = f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Local STT Adapter</title>
      <style>
        body {{
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                       "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
          max-width: 900px;
          margin: 40px auto;
          padding: 0 16px;
          color: #222;
          line-height: 1.65;
        }}
        h1 {{ margin-bottom: 8px; }}
        .card {{
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          padding: 18px;
          margin: 16px 0;
          background: #fafafa;
        }}
        code, pre {{
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        pre {{
          background: #111827;
          color: #f9fafb;
          padding: 14px;
          border-radius: 10px;
          overflow-x: auto;
        }}
        .muted {{ color: #666; }}
        ul {{ padding-left: 20px; }}
      </style>
    </head>
    <body>
      <h1>Local STT Adapter</h1>
      <p class="muted">纯本地语音转写适配服务，兼容 OpenAI 风格的 <code>/v1/audio/transcriptions</code> 接口。</p>

      <div class="card">
        <h2>当前配置</h2>
        <ul>
          <li><strong>模型</strong>：{MODEL_SIZE}</li>
          <li><strong>设备</strong>：{DEVICE}</li>
          <li><strong>计算类型</strong>：{COMPUTE_TYPE}</li>
          <li><strong>默认语言</strong>：{DEFAULT_LANGUAGE}</li>
          <li><strong>鉴权状态</strong>：{token_hint}</li>
        </ul>
      </div>

      <div class="card">
        <h2>接口说明</h2>
        <ul>
          <li><code>GET /</code>：默认说明页</li>
          <li><code>GET /healthz</code>：健康检查</li>
          <li><code>POST /v1/audio/transcriptions</code>：音频转写接口</li>
        </ul>
      </div>

      <div class="card">
        <h2>健康检查示例</h2>
        <pre>curl http://127.0.0.1:8080/healthz \\
  -H "Authorization: Bearer 你的Token"</pre>
      </div>

      <div class="card">
        <h2>转写示例</h2>
        <pre>curl -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \\
  -H "Authorization: Bearer 你的Token" \\
  -F "file=@test.wav" \\
  -F "language=zh"</pre>
      </div>

      <div class="card">
        <h2>OpenClaw 对接示例</h2>
        <pre>openclaw config set tools.media.audio.enabled 'true'
openclaw config set tools.media.audio.maxBytes '20971520'
openclaw config set tools.media.audio.timeoutSeconds '120'
openclaw config set tools.media.audio.echoTranscript 'false'
openclaw config set tools.media.audio.baseUrl '"http://127.0.0.1:8080/v1"'
openclaw config set tools.media.audio.headers '{{"Authorization":"Bearer 你的Token"}}'
openclaw config set tools.media.audio.models '[{{"provider":"openai","model":"whisper-1"}}]'</pre>
      </div>

      <div class="card">
        <h2>支持格式</h2>
        <p>常见音频格式如 <code>wav</code>、<code>mp3</code>、<code>ogg</code>、<code>m4a</code> 都可尝试上传。</p>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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
