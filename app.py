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
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="Local STT Adapter")

MODEL_SIZE = os.getenv("WHISPER_MODEL") or "small"
DEVICE = os.getenv("WHISPER_DEVICE") or "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE") or "int8"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE") or "zh"
API_TOKEN = (os.getenv("API_TOKEN") or "").strip()

AUDIO_PREVIEW_DIR = Path("/tmp/local-stt-audio-preview")
AUDIO_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

app.mount("/preview", StaticFiles(directory=str(AUDIO_PREVIEW_DIR)), name="preview")


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
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


def synthesize_text_to_wav(text: str, language: Optional[str]) -> str:
    lang = (language or DEFAULT_LANGUAGE or "zh").lower()

    voice_map = {
        "zh": "cmn",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
    }
    voice = voice_map.get(lang, "en")

    raw_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    raw_wav.close()

    final_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    final_wav.close()

    try:
        subprocess.run(
            [
                "espeak-ng",
                "-v",
                voice,
                "-s",
                "150",
                "-w",
                raw_wav.name,
                text,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_wav.name,
                "-ar",
                "16000",
                "-ac",
                "1",
                final_wav.name,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return final_wav.name
    finally:
        try:
            os.remove(raw_wav.name)
        except OSError:
            pass


def save_preview_audio(source_path: str) -> str:
    cleanup_old_preview_files()
    target_name = f"{uuid.uuid4().hex}.wav"
    target_path = AUDIO_PREVIEW_DIR / target_name
    shutil.copy2(source_path, target_path)
    return f"/preview/{target_name}"


def render_index(result_html: str = "", error_html: str = "") -> str:
    token_hint = "已启用 Bearer Token 鉴权" if API_TOKEN else "未启用鉴权"
    return f"""
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
          max-width: 960px;
          margin: 40px auto;
          padding: 0 16px;
          color: #222;
          line-height: 1.65;
          background: #f7f7f8;
        }}
        h1 {{ margin-bottom: 8px; }}
        .card {{
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 18px;
          margin: 16px 0;
          background: #fff;
          box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        code, pre, textarea {{
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        pre {{
          background: #111827;
          color: #f9fafb;
          padding: 14px;
          border-radius: 10px;
          overflow-x: auto;
          white-space: pre-wrap;
          word-break: break-word;
        }}
        .muted {{ color: #666; }}
        ul {{ padding-left: 20px; }}
        label {{
          display: block;
          font-weight: 600;
          margin: 12px 0 6px;
        }}
        input[type="text"], input[type="password"], select, textarea {{
          width: 100%;
          box-sizing: border-box;
          border: 1px solid #d1d5db;
          border-radius: 10px;
          padding: 10px 12px;
          font-size: 14px;
          background: #fff;
        }}
        textarea {{
          min-height: 110px;
          resize: vertical;
        }}
        input[type="file"] {{
          display: block;
          margin-top: 8px;
        }}
        button {{
          margin-top: 16px;
          border: 0;
          border-radius: 10px;
          padding: 10px 18px;
          font-size: 14px;
          font-weight: 600;
          background: #2563eb;
          color: #fff;
          cursor: pointer;
        }}
        button:hover {{
          background: #1d4ed8;
        }}
        .result {{
          background: #f9fafb;
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          padding: 14px;
        }}
        .error {{
          background: #fef2f2;
          border: 1px solid #fecaca;
          color: #991b1b;
          border-radius: 10px;
          padding: 14px;
        }}
        audio {{
          width: 100%;
          margin-top: 8px;
        }}
      </style>
    </head>
    <body>
      <h1>Local STT Adapter</h1>
      <p class="muted">纯本地语音转写适配服务，兼容 OpenAI 风格的 <code>/v1/audio/transcriptions</code> 接口。</p>

      <div class="card">
        <h2>当前配置</h2>
        <ul>
          <li><strong>模型</strong>：{html.escape(MODEL_SIZE)}</li>
          <li><strong>设备</strong>：{html.escape(DEVICE)}</li>
          <li><strong>计算类型</strong>：{html.escape(COMPUTE_TYPE)}</li>
          <li><strong>默认语言</strong>：{html.escape(DEFAULT_LANGUAGE)}</li>
          <li><strong>鉴权状态</strong>：{token_hint}</li>
        </ul>
      </div>

      <div class="card">
        <h2>在线测试：上传音频</h2>
        <form action="/test" method="post" enctype="multipart/form-data">
          <label for="token">Bearer Token</label>
          <input id="token" name="token" type="password" placeholder="输入你的 API Token" />

          <label for="language">语言</label>
          <select id="language" name="language">
            <option value="">自动/默认</option>
            <option value="zh" selected>中文 zh</option>
            <option value="en">英文 en</option>
            <option value="ja">日文 ja</option>
            <option value="ko">韩文 ko</option>
          </select>

          <label for="audio_file">音频文件</label>
          <input id="audio_file" name="audio_file" type="file" accept=".wav,.mp3,.ogg,.m4a,.aac,.flac,.webm" required />

          <button type="submit">上传并转写</button>
        </form>
      </div>

      <div class="card">
        <h2>在线测试：文字生成语音并转写</h2>
        <form action="/test-text" method="post">
          <label for="token2">Bearer Token</label>
          <input id="token2" name="token" type="password" placeholder="输入你的 API Token" />

          <label for="language2">语言</label>
          <select id="language2" name="language">
            <option value="zh" selected>中文 zh</option>
            <option value="en">英文 en</option>
            <option value="ja">日文 ja</option>
            <option value="ko">韩文 ko</option>
          </select>

          <label for="text">测试文本</label>
          <textarea id="text" name="text" placeholder="输入你想生成语音并测试转写的内容，例如：你好，这是本地语音识别测试。"></textarea>

          <button type="submit">生成语音、播放并转写</button>
        </form>
      </div>

      {error_html}
      {result_html}

      <div class="card">
        <h2>接口说明</h2>
        <ul>
          <li><code>GET /</code>：默认说明页</li>
          <li><code>GET /healthz</code>：健康检查</li>
          <li><code>POST /v1/audio/transcriptions</code>：音频转写接口</li>
          <li><code>POST /test</code>：网页上传测试</li>
          <li><code>POST /test-text</code>：网页文字生成测试语音并转写</li>
          <li><code>GET /preview/&lt;file&gt;</code>：试听生成音频</li>
        </ul>
      </div>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(render_index())


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


@app.post("/test", response_class=HTMLResponse)
async def test_page(
    token: str = Form(""),
    language: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
):
    try:
        auth_header = f"Bearer {token.strip()}" if token.strip() else None
        check_auth(auth_header)

        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="上传的文件为空")

        suffix = os.path.splitext(audio_file.filename or "audio.ogg")[1] or ".ogg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            text, info = transcribe_file(tmp_path, language=language)
            result_html = f"""
            <div class="card">
              <h2>上传测试结果</h2>
              <div class="result">
                <p><strong>文件名：</strong>{html.escape(audio_file.filename or "")}</p>
                <p><strong>识别语言：</strong>{html.escape(str(info.language))}</p>
                <p><strong>音频时长：</strong>{html.escape(str(info.duration))}</p>
                <p><strong>文本结果：</strong></p>
                <pre>{html.escape(text or "(空结果)")}</pre>
              </div>
            </div>
            """
            return HTMLResponse(render_index(result_html=result_html))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    except HTTPException as e:
        error_html = f"""
        <div class="card">
          <h2>上传测试失败</h2>
          <div class="error"><strong>错误：</strong>{html.escape(str(e.detail))}</div>
        </div>
        """
        return HTMLResponse(render_index(error_html=error_html), status_code=e.status_code)

    except Exception as e:
        error_html = f"""
        <div class="card">
          <h2>上传测试失败</h2>
          <div class="error"><strong>异常：</strong>{html.escape(str(e))}</div>
        </div>
        """
        return HTMLResponse(render_index(error_html=error_html), status_code=500)


@app.post("/test-text", response_class=HTMLResponse)
async def test_text_page(
    token: str = Form(""),
    language: Optional[str] = Form(None),
    text: str = Form(""),
):
    try:
        auth_header = f"Bearer {token.strip()}" if token.strip() else None
        check_auth(auth_header)

        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="请输入测试文本")

        wav_path = synthesize_text_to_wav(text, language=language)
        preview_url = save_preview_audio(wav_path)

        try:
            stt_text, info = transcribe_file(wav_path, language=language)

            result_html = f"""
            <div class="card">
              <h2>文字生成语音测试结果</h2>
              <div class="result">
                <p><strong>原始文本：</strong></p>
                <pre>{html.escape(text)}</pre>

                <p><strong>生成语音试听：</strong></p>
                <audio controls preload="metadata">
                  <source src="{html.escape(preview_url)}" type="audio/wav">
                  你的浏览器不支持音频播放。
                </audio>

                <p><strong>识别语言：</strong>{html.escape(str(info.language))}</p>
                <p><strong>音频时长：</strong>{html.escape(str(info.duration))}</p>
                <p><strong>转写结果：</strong></p>
                <pre>{html.escape(stt_text or "(空结果)")}</pre>
              </div>
            </div>
            """
            return HTMLResponse(render_index(result_html=result_html))
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    except HTTPException as e:
        error_html = f"""
        <div class="card">
          <h2>文字测试失败</h2>
          <div class="error"><strong>错误：</strong>{html.escape(str(e.detail))}</div>
        </div>
        """
        return HTMLResponse(render_index(error_html=error_html), status_code=e.status_code)

    except Exception as e:
        error_html = f"""
        <div class="card">
          <h2>文字测试失败</h2>
          <div class="error"><strong>异常：</strong>{html.escape(str(e))}</div>
        </div>
        """
        return HTMLResponse(render_index(error_html=error_html), status_code=500)
