"""
Whisper Worker

负责：
从 Redis 队列读取任务
执行 faster-whisper 识别
"""

import os
import redis
from faster_whisper import WhisperModel

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

STREAM = "stt_jobs"

redis_conn = redis.from_url(REDIS_URL)

model = WhisperModel(
    os.getenv("WHISPER_MODEL", "small"),
    device=os.getenv("WHISPER_DEVICE", "auto"),
    compute_type="int8"
)

print("Whisper Worker 已启动")

while True:

    jobs = redis_conn.xread({STREAM: "0"}, block=0)

    for stream, messages in jobs:

        for msg_id, data in messages:

            path = data[b"file"].decode()

            language = data[b"lang"].decode()

            segments, info = model.transcribe(
                path,
                language=language or None,
                beam_size=5,
                vad_filter=True
            )

            text = "".join(seg.text for seg in segments)

            print("识别结果:", text)

            os.remove(path)

            redis_conn.xdel(STREAM, msg_id)
