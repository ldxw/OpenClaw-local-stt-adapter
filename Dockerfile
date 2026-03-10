import html
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

DOCS_BASE_PATH = (os.getenv("DOCS_BASE_PATH") or "").strip()
if DOCS_BASE_PATH:
    if not DOCS_BASE_PATH.startswith("/"):
        DOCS_BASE_PATH = "/" + DOCS_BASE_PATH
    DOCS_BASE_PATH = DOCS_BASE_PATH.rstrip("/")
else:
    DOCS_BASE_PATH = ""

MODEL_SIZE = os.getenv("WHISPER_MODEL") or "small"
DEVICE = os.getenv("WHISPER_DEVICE") or "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or "int8"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or "zh"

API_TOKEN = (os.getenv("API_TOKEN") or "").strip()
API_TOKENS_RAW = os.getenv("API_TOKENS") or ""
API_TOKENS = {x.strip() for x in API_TOKENS_RAW.split(",") if x.strip()}
if API_TOKEN:
    API_TOKENS.add(API_TOKEN)

CORS_ALLOW_ORIGINS_RAW = os.getenv("CORS_ALLOW_ORIGINS") or ""
CORS_ALLOW_ORIGINS = [x.strip() for x in CORS_ALLOW_ORIGINS_RAW.split(",") if x.strip()]
ALLOW_ALL_ORIGINS = "*" in CORS_ALLOW_ORIGINS

app = FastAPI(title="Local STT Adapter")

if ALLOW_ALL_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

AUDIO_PREVIEW_DIR = Path("/tmp/local-stt-audio-preview")
AUDIO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

app.mount("/preview", StaticFiles(directory=str(AUDIO_PREVIEW_DIR)), name="preview")


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKENS:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="unauthorized")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in API_TOKENS:
        raise HTTPException(status_code=401, detail="unauthorized")


def cleanup_old_preview_files(max_age_seconds: int = 3600) -> None:
    now = time.time()
    for path in AUDIO_PREVIEW_DIR.glob("*"):
        try:
            if path.is_file() and now - path.stat().st_mtime > max_age_seconds:
                path.unlink(missing_ok=True)
        except OSError:
            pass


def transcribe_file(path: str, language: Optional[str], prompt: Optional[str] = None):
    segments, info = model.transcribe(
        path,
        language=language or DEFAULT_LANGUAGE,
        initial_prompt=prompt,
        vad_filter=True,
        beam_size=5,
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, info


def render_docs_page() -> str:
    token_hint = "已启用 Bearer Token 鉴权" if API_TOKENS else "未启用鉴权"
    cors_hint = "未配置"
    if ALLOW_ALL_ORIGINS:
        cors_hint = "允许全部来源 *"
    elif CORS_ALLOW_ORIGINS:
        cors_hint = ", ".join(CORS_ALLOW_ORIGINS)

    docs_path_display = (DOCS_BASE_PATH + "/") if DOCS_BASE_PATH else "/"

    return f"""
    <!doctype html>
    <html lang="zh-CN">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Local STT Adapter Docs</title>
      <style>
        :root {{
          --bg: #f5f7fb;
          --card: rgba(255,255,255,0.9);
          --text: #172033;
          --muted: #667085;
          --line: #e6eaf2;
          --primary: #3b82f6;
          --primary-hover: #2563eb;
          --secondary: #374151;
          --secondary-hover: #1f2937;
          --shadow: 0 8px 32px rgba(17, 24, 39, 0.08);
          --radius: 18px;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC",
                       "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
          color: var(--text);
          background:
            radial-gradient(circle at top left, rgba(59,130,246,0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(16,185,129,0.06), transparent 24%),
            var(--bg);
        }}
        .container {{
          max-width: 1120px;
          margin: 0 auto;
          padding: 28px 16px 48px;
        }}
        .hero {{
          overflow: hidden;
          border: 1px solid rgba(255,255,255,0.55);
          background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(248,250,252,0.88));
          box-shadow: var(--shadow);
          border-radius: 24px;
          padding: 28px 24px;
          margin-bottom: 20px;
        }}
        .hero h1 {{
          margin: 0 0 8px;
          font-size: 32px;
          line-height: 1.15;
          letter-spacing: -0.02em;
        }}
        .hero p {{
          margin: 0;
          color: var(--muted);
          font-size: 15px;
        }}
        .grid {{
          display: grid;
          grid-template-columns: 1.1fr 0.9fr;
          gap: 18px;
        }}
        .stack {{
          display: grid;
          gap: 18px;
        }}
        .card {{
          border: 1px solid rgba(230,234,242,0.95);
          border-radius: var(--radius);
          padding: 20px;
          background: var(--card);
          box-shadow: var(--shadow);
        }}
        .card h2 {{
          margin: 0 0 10px;
          font-size: 18px;
        }}
        .section-desc {{
          margin: 0;
          color: var(--muted);
          font-size: 14px;
        }}
        .meta-list {{
          margin: 0;
          padding-left: 18px;
        }}
        .meta-list li {{
          margin: 6px 0;
        }}
        pre {{
          margin: 10px 0 0;
          background: #0f172a;
          color: #f8fafc;
          padding: 14px;
          border-radius: 14px;
          overflow-x: auto;
          white-space: pre-wrap;
          word-break: break-word;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
          line-height: 1.55;
        }}
        button {{
          border: 0;
          border-radius: 12px;
          padding: 11px 16px;
          font-size: 14px;
          font-weight: 700;
          background: var(--secondary);
          color: #fff;
          cursor: pointer;
          margin-top: 10px;
        }}
        button:hover {{
          background: var(--secondary-hover);
        }}
        details {{
          margin-top: 12px;
          border: 1px solid #e5eaf3;
          border-radius: 14px;
          background: rgba(250,251,253,0.92);
          overflow: hidden;
        }}
        summary {{
          cursor: pointer;
          font-weight: 700;
          padding: 14px 16px;
          user-select: none;
          list-style: none;
        }}
        .details-body {{
          padding: 0 16px 16px;
        }}
        @media (max-width: 900px) {{
          .grid {{
            grid-template-columns: 1fr;
          }}
          .hero h1 {{
            font-size: 26px;
          }}
        }}
      </style>
    </head>
    <body>
      <div class="container">
        <section class="hero">
          <h1>Local STT Adapter Docs</h1>
          <p>公开说明页。仅展示接口说明、配置状态和对接示例，不再提供在线验证或在线测试。</p>
        </section>

        <div class="grid">
          <div class="stack">
            <div class="card">
              <h2>当前配置</h2>
              <p class="section-desc">这里显示当前服务运行参数。</p>
              <ul class="meta-list">
                <li><strong>模型：</strong>{html.escape(MODEL_SIZE)}</li>
                <li><strong>设备：</strong>{html.escape(DEVICE)}</li>
                <li><strong>计算类型：</strong>{html.escape(COMPUTE_TYPE)}</li>
                <li><strong>默认语言：</strong>{html.escape(DEFAULT_LANGUAGE)}</li>
                <li><strong>鉴权状态：</strong>{token_hint}</li>
                <li><strong>多密钥数量：</strong>{len(API_TOKENS)}</li>
                <li><strong>CORS 允许来源：</strong>{html.escape(cors_hint)}</li>
                <li><strong>说明页目录：</strong>{html.escape(docs_path_display)}</li>
                <li><strong>接口根路径：</strong>/</li>
              </ul>
            </div>

            <div class="card">
              <h2>接口说明</h2>
              <p class="section-desc">接口始终保持原始路径，不受说明页目录影响。</p>

              <h3 style="margin:18px 0 6px;">健康检查</h3>
              <pre id="healthCmd">curl "__BASE_ORIGIN__/healthz" \\
  -H "Authorization: Bearer 你的Token"</pre>
              <button type="button" id="copyHealthBtn">复制健康检查命令</button>

              <h3 style="margin:18px 0 6px;">音频转写</h3>
              <pre id="transcribeCmd">curl -X POST "__BASE_ORIGIN__/v1/audio/transcriptions" \\
  -H "Authorization: Bearer 你的Token" \\
  -F "file=@test.wav" \\
  -F "language=zh"</pre>
              <button type="button" id="copyTranscribeBtn">复制转写命令</button>
            </div>
          </div>

          <div class="stack">
            <div class="card">
              <h2>对接示例</h2>

              <details>
                <summary>查看 OpenClaw 对接示例</summary>
                <div class="details-body">
                  <pre id="openclawCmd">openclaw config set tools.media.audio.enabled 'true'
openclaw config set tools.media.audio.maxBytes '20971520'
openclaw config set tools.media.audio.timeoutSeconds '120'
openclaw config set tools.media.audio.echoTranscript 'false'
openclaw config set tools.media.audio.baseUrl '"__BASE_ORIGIN__/v1"'
openclaw config set tools.media.audio.headers '{{"Authorization":"Bearer 你的Token"}}'
openclaw config set tools.media.audio.models '[{{"provider":"openai","model":"whisper-1"}}]'</pre>
                </div>
              </details>

              <details>
                <summary>查看 TTS 推荐策略</summary>
                <div class="details-body">
                  <pre>openclaw config set messages.tts.auto '"inbound"'
openclaw config set messages.tts.mode '"final"'
openclaw config set messages.tts.provider '"edge"'
openclaw config set messages.tts.modelOverrides.enabled 'false'
openclaw config set messages.tts.edge.enabled 'true'
openclaw config set messages.tts.edge.voice '"zh-CN-XiaoxiaoNeural"'</pre>
                </div>
              </details>
            </div>
          </div>
        </div>
      </div>

      <script>
        function copyText(text) {{
          navigator.clipboard.writeText(text).catch(() => {{}});
        }}

        function buildDynamicCommands() {{
          const base = window.location.origin;

          document.getElementById("healthCmd").innerText =
`curl "${{base}}/healthz" \\
  -H "Authorization: Bearer 你的Token"`;

          document.getElementById("transcribeCmd").innerText =
`curl -X POST "${{base}}/v1/audio/transcriptions" \\
  -H "Authorization: Bearer 你的Token" \\
  -F "file=@test.wav" \\
  -F "language=zh"`;

          document.getElementById("openclawCmd").innerText =
`openclaw config set tools.media.audio.enabled 'true'
openclaw config set tools.media.audio.maxBytes '20971520'
openclaw config set tools.media.audio.timeoutSeconds '120'
openclaw config set tools.media.audio.echoTranscript 'false'
openclaw config set tools.media.audio.baseUrl '"${{base}}/v1"'
openclaw config set tools.media.audio.headers '{{"Authorization":"Bearer 你的Token"}}'
openclaw config set tools.media.audio.models '[{{"provider":"openai","model":"whisper-1"}}]'`;
        }}

        window.addEventListener("DOMContentLoaded", function () {{
          buildDynamicCommands();
          document.getElementById("copyHealthBtn").addEventListener("click", function () {{
            copyText(document.getElementById("healthCmd").innerText);
          }});
          document.getElementById("copyTranscribeBtn").addEventListener("click", function () {{
            copyText(document.getElementById("transcribeCmd").innerText);
          }});
        }});
      </script>
    </body>
    </html>
    """


if DOCS_BASE_PATH:
    @app.get("/")
    async def root_page():
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    @app.get(DOCS_BASE_PATH, include_in_schema=False)
    async def redirect_docs():
        return RedirectResponse(url=f"{DOCS_BASE_PATH}/")

    @app.get(f"{DOCS_BASE_PATH}/", response_class=HTMLResponse)
    async def docs_page():
        return HTMLResponse(render_docs_page())
else:
    @app.get("/", response_class=HTMLResponse)
    async def docs_page_root():
        return HTMLResponse(render_docs_page())


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
        text, info = transcribe_file(tmp_path, language=language, prompt=prompt)
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
