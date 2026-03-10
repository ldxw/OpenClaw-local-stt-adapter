"""
Whisper STT API 服务
用于接收语音并发送到 Redis 队列

兼容：
OpenAI API
OpenClaw
小龙虾
QQBot
"""

import os
import uuid
import time
import redis

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import HTMLResponse


# Redis 地址
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

# API Token（可选）
API_TOKEN = os.getenv("API_TOKEN", "")

# Redis 连接
redis_conn = redis.from_url(REDIS_URL)

# Redis Stream 队列名称
STREAM = "stt_jobs"

# 启动时间
START_TIME = time.time()

# FastAPI 实例
app = FastAPI(title="Whisper STT API")


def check_auth(auth):
    """
    检查 API Token
    """

    if not API_TOKEN:
        return

    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401)

    token = auth.replace("Bearer ", "")

    if token != API_TOKEN:
        raise HTTPException(401)


@app.get("/healthz")
def health():
    """
    健康检查接口
    用于 Docker / 监控检测
    """

    try:
        redis_conn.ping()
        redis_status = "ok"
    except:
        redis_status = "error"

    queue_len = redis_conn.xlen(STREAM)

    uptime = int(time.time() - START_TIME)

    return {
        "status": "ok",
        "redis": redis_status,
        "queue": queue_len,
        "uptime": uptime
    }


@app.get("/v1/models")
def models():
    """
    返回支持的模型
    OpenAI API 兼容
    """

    return {
        "object": "list",
        "data": [
            {"id": "whisper-1"},
            {"id": "gpt-4o-mini-transcribe"},
            {"id": "gpt-4o-transcribe"}
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
        file: UploadFile = File(...),
        language: str = Form(None),
        authorization: str = Header(None)
):
    """
    语音识别接口
    """

    check_auth(authorization)

    # 读取音频
    content = await file.read()

    # 生成任务ID
    job_id = str(uuid.uuid4())

    # 保存临时文件
    path = f"/tmp/{job_id}.audio"

    with open(path, "wb") as f:
        f.write(content)

    # 写入 Redis 队列
    redis_conn.xadd(
        STREAM,
        {
            "id": job_id,
            "file": path,
            "lang": language or ""
        }
    )

    return {"job_id": job_id}


@app.get("/docs", response_class=HTMLResponse)
def docs():
    """
    返回 API 文档页面
    """

    with open("docs/docs.html") as f:
        return f.read()
