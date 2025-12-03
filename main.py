
import os
import wave
import asyncio
from cache import AudioCache
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from starlette.responses import HTMLResponse
from contextlib import asynccontextmanager

from starlette.websockets import WebSocketState

from config import asr_config
from model_serve import asr_model
from model_serve import VADStreamProcessor

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 加载模型
    print("开始加载模型")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", '0')
    asr_model.load_model(devices)
    yield  # fastapi service


app = FastAPI(lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def recognize_audio_file(file: UploadFile = File(...)):
    content = await file.read()
    with asr_model.recognize(stream=False) as model:
        asr_res = model.audio_recognition(content)
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
    vad_processor = VADStreamProcessor()
    with asr_model.recognize(stream=True) as model:
        try:
            while True:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5)
                audio_cache.append_meta(data, vad_processor.accept)
                for pcm_buf in audio_cache:
                    text = model.audio_stream_recognition(pcm_buf)
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await asyncio.wait_for(websocket.send_json(
                            {"text": text.replace(' ', ''), "is_final": audio_cache.final}), timeout=5)
        except asyncio.TimeoutError:
            print("websocket timeout")

@app.get("/health")
async def health():
    """
    模型正常加载
    :return:
    """
    if not asr_model.n_zh and not asr_model.n_stream:
        return HTTPException(status_code=500, detail="未成功加载ASR模型")
    return {"models": f"paraformer-zh: {asr_model.zh_model.n_gpu}, paraformer-zh-stream: {asr_model.stream_model.n_gpu}"}

@app.get("/test")
async def test():
    '''读取本地wav文件，切片调用定义的函数debug'''
    with wave.open("socket.wav", "rb") as f:
        audio_bytes = f.readframes(f.getnframes())
    # 切分成多份，然后模拟socket返回给到函数处理
    chunk_size = asr_config.chunk_size_bits
    with asr_model.recognize(stream=True) as model:
        while len(audio_bytes) >= chunk_size:
            buffer = audio_bytes[:chunk_size]
            text = model.audio_stream_recognition(buffer)
            audio_bytes = audio_bytes[chunk_size:]
        audio_bytes += b'\x00' * (chunk_size - len(audio_bytes))
        print(model.audio_stream_recognition(audio_bytes, is_final=True))


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
