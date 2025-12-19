import asyncio
import tempfile
import os
import time
import subprocess
from cache import AudioCache
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from starlette.responses import HTMLResponse
from contextlib import asynccontextmanager

from starlette.websockets import WebSocketDisconnect

from model_serve import asr_model
from audio_vad import AudioVAD

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 加载模型
    print("开始加载模型")
    asr_model.load_model()
    yield  # fastapi service


app = FastAPI(lifespan=lifespan)


def convert_audio_to_pcm(data: bytes, temp_out, sr: int = 16000, ac: int = 1):
    """
    自动识别音频格式，将任意音频 bytes 转换为 pcm_s16le 格式。
    - 已经是PCM16 不做处理
    """

    if data.startswith(b"RIFF"):
        input_format = "wav"
    elif data[4:8] == b"ftyp":
        input_format = "mp4"  # m4a/3gp 都是 MP4 容器
    elif data.startswith(b"ID3"):
        input_format = "mp3"
    elif data[:4] in (b"fLaC", b"OggS"):
        input_format = "flac" if data[:4] == b"fLaC" else "ogg"
    elif len(data) % 2 == 0 and all(abs(b-128) < 128 for b in data[:100]):
        # PCM 裸流
        input_format = "s16le"
    else:
        # 默认尝试 mp3
        input_format = "mp3"
    if input_format.lower() in {"s16le", "pcm_s16le"}:
        temp_out.write(data)
        temp_out.close()
        return
    cleanup_paths = []
    try:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}")
        tmp_in.write(data)
        tmp_in.close()
        temp_out.close()
        cleanup_paths = [tmp_in.name]

        cmd = [
            "ffmpeg", "-y", "-i", tmp_in.name,
            "-ac", str(ac), "-ar", str(sr),
            "-acodec", "pcm_s16le", temp_out.name,
            "-loglevel", "error"
        ]
        proc = subprocess.run(cmd, capture_output=True)

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed ({input_format})\n{proc.stderr.decode()}")
    finally:
        # --- 清理临时文件 ---
        for path in cleanup_paths:
            for _ in range(10):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                    break
                except PermissionError:
                    time.sleep(0.05)
            else:
                print(f"⚠️ Warning: could not remove temporary file {path}")


@app.post("/v1/audio/transcriptions")
async def recognize_audio_file(file: UploadFile = File(...)):
    """
    将上传的文件转成PCM格式，存放到临时文件
    后续模型要识别基于临时文件来做
    :param file:
    :return:
    """
    content = await file.read()
    try:
        with asr_model.recognize(stream=False) as model:
            convert_audio_to_pcm(content, model.temp_file)
            asr_res = model.audio_recognition()
    except Exception as e:
        print(e)
        asr_res = "服务端异常，请稍等重试"
        return HTTPException(status_code=500, detail=asr_res)
    return {
        "text": asr_res,
        "usage": {
            "type": "tokens",
            "input_tokens": 14,
            "input_token_details": {
                "text_tokens": 0,
                "audio_tokens": 14
            },
            "output_tokens": 45,
            "total_tokens": 59
        }
    }

@app.websocket("/ws/audio")
async def recognize_audio_stream(websocket: WebSocket):
    """
    音频流式处理 10帧识别一次  10 * 60ms
    2个字节长度 = 1个采样点
    :param websocket:
    :return:
    """
    #数据初始化
    await websocket.accept()
    audio_cache = AudioCache()
    audio_vad = AudioVAD(vad_mode=3)
    with asr_model.recognize(stream=True) as model:
        try:
            while True:
                data = await websocket.receive_bytes()
                audio_cache.append_meta(data, audio_vad.stream_vad)
                for pcm_buf, final in audio_cache:
                    text = model.audio_stream_recognition(pcm_buf).replace(' ', '')
                    if len(text) > 0:
                        await websocket.send_json({"text": text,"is_final": False})
                    if final:
                        await websocket.send_json({"text": '', "is_final": True})
                        await websocket.close()
                        return
        except asyncio.TimeoutError:
            print("websocket timeout")
        except WebSocketDisconnect:
            print("websocket disconnect")

@app.get("/health")
async def health():
    """
    模型正常加载
    :return:
    """
    if not asr_model.n_zh and not asr_model.n_stream:
        return HTTPException(status_code=500, detail="未成功加载ASR模型")
    return {"models": f"paraformer-zh: {asr_model.zh_model.n_gpu}, paraformer-zh-stream: {asr_model.stream_model.n_gpu}"}

import struct
@app.websocket("/ws/test")
async def test(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        flag = struct.pack("I", data[:4])
        print("data :", data.decode())
        await websocket.send_bytes(b'alive')
        if flag == 1:
            await websocket.close()


# 用于测试的简单 HTML 页面（包含录音和连接逻辑）
@app.get("/")
async def get():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>Audio Streaming via WebSocket</title></head>
<body>
    <h2>录音并通过 WebSocket 发送到 FastAPI</h2>
    <button id="start">开始录音</button>
    <button id="stop">停止录音</button>
    <p id="status">状态: 未连接</p>

    <script>
        let mediaRecorder;
        let ws;

        async function startRecording() {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const socketUrl = `ws://${window.location.host}/ws/audio`;
            ws = new WebSocket(socketUrl);

            ws.onopen = () => {
                document.getElementById('status').innerText = '状态: 已连接 WebSocket';
            };

            ws.onerror = (error) => {
                console.error('WebSocket 错误:', error);
                document.getElementById('status').innerText = '状态: 连接失败';
            };

            // 使用 MediaRecorder 录制音频（WebM + Opus）
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                    ws.send(event.data); // 发送二进制音频块
                }
            };

            mediaRecorder.start(1000); // 每1秒发送一块音频
            document.getElementById('status').innerText = '状态: 录音中...';
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            if (ws) {
                ws.close();
            }
            document.getElementById('status').innerText = '状态: 已停止';
        }

        document.getElementById('start').onclick = startRecording;
        document.getElementById('stop').onclick = stopRecording;
    </script>
</body>
</html>
    """)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400)
