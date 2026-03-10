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

DEFAULT_TEST_TEXT = "你好，这是本地语音识别测试。现在正在验证文字生成语音、播放和转写链路是否正常。"

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


def synthesize_text_to_wav(text: str, language: Optional[str], voice_gender: str = "female") -> str:
    lang = (language or DEFAULT_LANGUAGE or "zh").lower()

    voice_map = {
        "zh": {"female": "cmn", "male": "cmn"},
        "en": {"female": "en-us+f3", "male": "en-us+m3"},
        "ja": {"female": "ja", "male": "ja"},
        "ko": {"female": "ko", "male": "ko"},
    }

    lang_map = voice_map.get(lang, voice_map["en"])
    voice = lang_map.get(voice_gender, lang_map["female"])

    speed = "150" if voice_gender == "female" else "145"
    pitch = "50" if voice_gender == "female" else "35"

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
                speed,
                "-p",
                pitch,
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


def render_index() -> str:
    token_hint = "已启用 Bearer Token 鉴权" if API_TOKEN else "未启用鉴权"
    default_text = html.escape(DEFAULT_TEST_TEXT)

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
          max-width: 1080px;
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
          min-height: 120px;
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
        .row {{
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }}
        .row > * {{
          flex: 1 1 auto;
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
        .success {{
          background: #f0fdf4;
          border: 1px solid #bbf7d0;
          color: #166534;
          border-radius: 10px;
          padding: 14px;
        }}
        .badge {{
          display: inline-block;
          font-size: 12px;
          padding: 4px 10px;
          border-radius: 999px;
          background: #e5e7eb;
          color: #111827;
          margin-left: 8px;
          vertical-align: middle;
        }}
        .badge.ok {{
          background: #dcfce7;
          color: #166534;
        }}
        audio {{
          width: 100%;
          margin-top: 8px;
        }}
        .hidden {{
          display: none;
        }}
        .copy-btn {{
          background: #374151;
        }}
        .copy-btn:hover {{
          background: #1f2937;
        }}
      </style>
    </head>
    <body>
      <h1>Local STT Adapter <span id="verifyBadge" class="badge">请先验证密钥</span></h1>
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
        <h2>第一步：验证密钥</h2>
        <form id="verifyForm">
          <label for="token">Token</label>
          <input id="token" name="token" type="password" placeholder="只填 token 本身，不要加 Bearer" />
          <div class="row">
            <button type="submit">验证密钥</button>
            <button type="button" id="clearSavedTokenBtn" class="copy-btn">清除本地保存密钥</button>
          </div>
        </form>
        <div id="verifyMessage"></div>
      </div>

      <div class="card">
        <h2>命令测试接口</h2>
        <p class="muted">这个区域始终显示，不需要先验证密钥。把下面命令中的 <code>你的Token</code> 替换成真实 token 即可。</p>
        <div class="row">
          <button type="button" id="copyHealthBtn" class="copy-btn">复制健康检查命令</button>
          <button type="button" id="copyTranscribeBtn" class="copy-btn">复制转写命令</button>
        </div>
        <pre id="healthCmd">curl "http://127.0.0.1:8080/healthz" \\
  -H "Authorization: Bearer 你的Token"</pre>
        <pre id="transcribeCmd">curl -X POST "http://127.0.0.1:8080/v1/audio/transcriptions" \\
  -H "Authorization: Bearer 你的Token" \\
  -F "file=@test.wav" \\
  -F "language=zh"</pre>
      </div>

      <div id="testingSection" class="hidden">
        <div class="card">
          <h2>在线测试：上传音频</h2>
          <form id="uploadTestForm">
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
          <form id="textTestForm">
            <label for="language2">语言</label>
            <select id="language2" name="language">
              <option value="zh" selected>中文 zh</option>
              <option value="en">英文 en</option>
              <option value="ja">日文 ja</option>
              <option value="ko">韩文 ko</option>
            </select>

            <label for="voice_gender">语音性别</label>
            <select id="voice_gender" name="voice_gender">
              <option value="female" selected>女声</option>
              <option value="male">男声</option>
            </select>

            <label for="text">测试文本</label>
            <textarea id="text" name="text">{default_text}</textarea>

            <div class="row">
              <button type="button" id="fillDefaultBtn" class="copy-btn">填入默认文字</button>
              <button type="submit">生成语音、播放并转写</button>
            </div>
          </form>
        </div>
      </div>

      <div id="resultBox"></div>

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

      <script>
        let verifiedToken = "";
        const defaultText = {html.escape(repr(DEFAULT_TEST_TEXT))};
        const TOKEN_STORAGE_KEY = "local_stt_adapter_token";

        const verifyForm = document.getElementById("verifyForm");
        const verifyMessage = document.getElementById("verifyMessage");
        const verifyBadge = document.getElementById("verifyBadge");
        const testingSection = document.getElementById("testingSection");
        const resultBox = document.getElementById("resultBox");
        const tokenInput = document.getElementById("token");

        function escapeHtml(str) {{
          return String(str)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
        }}

        function copyText(text) {{
          navigator.clipboard.writeText(text).then(() => {{
            verifyMessage.innerHTML = `<div class="success">命令已复制到剪贴板。</div>`;
          }}).catch((err) => {{
            verifyMessage.innerHTML = `<div class="error">复制失败：${{escapeHtml(err)}}</div>`;
          }});
        }}

        function setVerifySuccess(message) {{
          verifyBadge.textContent = "密钥已验证";
          verifyBadge.className = "badge ok";
          verifyMessage.innerHTML = `<div class="success">${{escapeHtml(message)}}</div>`;
          testingSection.classList.remove("hidden");
        }}

        function setVerifyError(message) {{
          verifyBadge.textContent = "请先验证密钥";
          verifyBadge.className = "badge";
          verifyMessage.innerHTML = `<div class="error">${{escapeHtml(message)}}</div>`;
          testingSection.classList.add("hidden");
          verifiedToken = "";
        }}

        function renderUploadResult(data) {{
          resultBox.innerHTML = `
            <div class="card">
              <h2>上传测试结果</h2>
              <div class="result">
                <p><strong>文件名：</strong>${{escapeHtml(data.filename || "")}}</p>
                <p><strong>识别语言：</strong>${{escapeHtml(data.language || "")}}</p>
                <p><strong>音频时长：</strong>${{escapeHtml(String(data.duration ?? ""))}}</p>
                <p><strong>文本结果：</strong></p>
                <pre>${{escapeHtml(data.text || "(空结果)")}}</pre>
              </div>
            </div>
          `;
        }}

        function renderTextResult(data) {{
          resultBox.innerHTML = `
            <div class="card">
              <h2>文字生成语音测试结果</h2>
              <div class="result">
                <p><strong>原始文本：</strong></p>
                <pre>${{escapeHtml(data.original_text || "")}}</pre>
                <p><strong>语音性别：</strong>${{escapeHtml(data.voice_gender_label || "")}}</p>
                <p><strong>生成语音试听：</strong></p>
                <audio controls preload="metadata">
                  <source src="${{escapeHtml(data.preview_url || "")}}" type="audio/wav">
                  你的浏览器不支持音频播放。
                </audio>
                <p><strong>识别语言：</strong>${{escapeHtml(data.language || "")}}</p>
                <p><strong>音频时长：</strong>${{escapeHtml(String(data.duration ?? ""))}}</p>
                <p><strong>转写结果：</strong></p>
                <pre>${{escapeHtml(data.text || "(空结果)")}}</pre>
              </div>
            </div>
          `;
        }}

        function renderError(title, message) {{
          resultBox.innerHTML = `
            <div class="card">
              <h2>${{escapeHtml(title)}}</h2>
              <div class="error">${{escapeHtml(message)}}</div>
            </div>
          `;
        }}

        async function verifyToken(token, silent = false) {{
          const formData = new FormData();
          formData.append("token", token);

          try {{
            const resp = await fetch("/verify-json", {{
              method: "POST",
              body: formData
            }});
            const data = await resp.json();

            if (!resp.ok || !data.ok) {{
              localStorage.removeItem(TOKEN_STORAGE_KEY);
              setVerifyError(data.detail || "验证失败");
              return false;
            }}

            verifiedToken = token;
            localStorage.setItem(TOKEN_STORAGE_KEY, token);
            setVerifySuccess(silent ? "已自动恢复并验证本地保存的密钥。" : "密钥验证通过，在线测试表单已开放。");
            return true;
          }} catch (err) {{
            setVerifyError("请求失败：" + err);
            return false;
          }}
        }}

        verifyForm.addEventListener("submit", async function (e) {{
          e.preventDefault();
          const token = tokenInput.value.trim();
          if (!token) {{
            setVerifyError("请输入 token");
            return;
          }}
          await verifyToken(token, false);
        }});

        document.getElementById("clearSavedTokenBtn").addEventListener("click", function () {{
          localStorage.removeItem(TOKEN_STORAGE_KEY);
          verifiedToken = "";
          tokenInput.value = "";
          verifyBadge.textContent = "请先验证密钥";
          verifyBadge.className = "badge";
          verifyMessage.innerHTML = `<div class="success">已清除本地保存的密钥。</div>`;
          testingSection.classList.add("hidden");
          resultBox.innerHTML = "";
        }});

        document.getElementById("copyHealthBtn").addEventListener("click", function () {{
          copyText(document.getElementById("healthCmd").innerText);
        }});

        document.getElementById("copyTranscribeBtn").addEventListener("click", function () {{
          copyText(document.getElementById("transcribeCmd").innerText);
        }});

        document.getElementById("uploadTestForm").addEventListener("submit", async function (e) {{
          e.preventDefault();
          if (!verifiedToken) {{
            renderError("上传测试失败", "请先验证密钥");
            return;
          }}

          const fileInput = document.getElementById("audio_file");
          if (!fileInput.files.length) {{
            renderError("上传测试失败", "请选择音频文件");
            return;
          }}

          const formData = new FormData();
          formData.append("token", verifiedToken);
          formData.append("language", document.getElementById("language").value);
          formData.append("audio_file", fileInput.files[0]);

          try {{
            const resp = await fetch("/test-json", {{
              method: "POST",
              body: formData
            }});
            const data = await resp.json();

            if (!resp.ok || !data.ok) {{
              renderError("上传测试失败", data.detail || "请求失败");
              return;
            }}

            renderUploadResult(data);
          }} catch (err) {{
            renderError("上传测试失败", "请求失败：" + err);
          }}
        }});

        document.getElementById("textTestForm").addEventListener("submit", async function (e) {{
          e.preventDefault();
          if (!verifiedToken) {{
            renderError("文字测试失败", "请先验证密钥");
            return;
          }}

          const formData = new FormData();
          formData.append("token", verifiedToken);
          formData.append("language", document.getElementById("language2").value);
          formData.append("voice_gender", document.getElementById("voice_gender").value);
          formData.append("text", document.getElementById("text").value);

          try {{
            const resp = await fetch("/test-text-json", {{
              method: "POST",
              body: formData
            }});
            const data = await resp.json();

            if (!resp.ok || !data.ok) {{
              renderError("文字测试失败", data.detail || "请求失败");
              return;
            }}

            renderTextResult(data);
          }} catch (err) {{
            renderError("文字测试失败", "请求失败：" + err);
          }}
        }});

        document.getElementById("fillDefaultBtn").addEventListener("click", function () {{
          document.getElementById("text").value = defaultText;
        }});

        window.addEventListener("DOMContentLoaded", async function () {{
          const savedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
          if (savedToken) {{
            tokenInput.value = savedToken;
            await verifyToken(savedToken, true);
          }}
        }});
      </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(render_index())


@app.post("/verify-json")
async def verify_json(token: str = Form("")):
    token = (token or "").strip()
    try:
        auth_header = f"Bearer {token}" if token else None
        check_auth(auth_header)
        return {"ok": True}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"ok": False, "detail": str(e.detail)})


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


@app.post("/test-json")
async def test_json(
    token: str = Form(""),
    language: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
):
    token = (token or "").strip()
    auth_header = f"Bearer {token}" if token else None
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
        return {
            "ok": True,
            "filename": audio_file.filename or "",
            "language": str(info.language),
            "duration": info.duration,
            "text": text,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.post("/test-text-json")
async def test_text_json(
    token: str = Form(""),
    language: Optional[str] = Form(None),
    voice_gender: str = Form("female"),
    text: str = Form(""),
):
    token = (token or "").strip()
    language = language or "zh"
    voice_gender = voice_gender or "female"

    auth_header = f"Bearer {token}" if token else None
    check_auth(auth_header)

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入测试文本")

    wav_path = synthesize_text_to_wav(text, language=language, voice_gender=voice_gender)
    preview_url = save_preview_audio(wav_path)

    try:
        stt_text, info = transcribe_file(wav_path, language=language)
        return {
            "ok": True,
            "original_text": text,
            "voice_gender": voice_gender,
            "voice_gender_label": "女声" if voice_gender == "female" else "男声",
            "preview_url": preview_url,
            "language": str(info.language),
            "duration": info.duration,
            "text": stt_text,
        }
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
