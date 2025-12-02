import json
import os
import wave

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
    # 后处理
    asr_model.clear()

app = FastAPI(lifespan=lifespan)

def save_audio_to_wav(audio_data: bytearray, filename: str = None):
    """
        将音频数据保存为WAV文件
        :param audio_data: 音频字节数据
        :param filename: 保存的文件名，
    """
    import datetime
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filepath = os.path.join("recorded_audios", filename)

    #创建WAV文件
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(1600)
        wf.writeframes(audio_data)

    print(f"保存文件：{filepath}")


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
    vad_processor = VADStreamProcessor()
    await websocket.accept()
    audio_buffer = bytearray()
    chunk_size = asr_config.chunk_size_bits
    # 用于保存音频数据的列表
    received_audio_data = bytearray()
    with asr_model.recognize(stream=True) as model:
        while True:
            message = await websocket.receive()
            # 客户端断开
            if message["type"] == "websocket.disconnect":
                save_audio_to_wav(audio_buffer)
                break
            # 接收到数据
            if message["type"] == "websocket.receive":
                # 二进制音频
                if "bytes" in message:
                    audio = vad_processor.accept(data=message["bytes"])
                    if audio is not None:
                        if audio[0][0] is not "":
                            audio_buffer.extend(audio[0][0])
                    while len(audio_buffer) >= chunk_size:
                        buffer = audio_buffer[:chunk_size]
                        raw = model.audio_stream_recognition(buffer)
                        # 规范化返回：支持 None / "" / [] / [""] 等
                        texts = []
                        if raw is None:
                            texts = []
                        elif isinstance(raw, (list, tuple)):
                            #过滤空字符
                            texts = [t for t in raw if t is not None and str(t).strip() != ""]
                        else:
                            s = str(raw)
                            if s.strip() != "":
                                texts = [s]
                        if texts:
                            text = "".join(texts)
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json({"text": text, "is_final": False})
                        audio_buffer = audio_buffer[chunk_size:]
                # 文本控制帧
                elif "text" in message:
                    try:
                        ctrl = json.loads(message["text"])
                        if ctrl.get("type") == "end":
                            save_audio_to_wav(received_audio_data)
                            audio_buffer += b'\x00' * (chunk_size - len(audio_buffer))
                            final_text = model.audio_stream_recognition_with_vad(audio_buffer, is_final=True)
                            if websocket.client_state == WebSocketState.CONNECTED:
                                await websocket.send_json({
                                    "text": final_text,
                                    "is_final": True
                                })
                            break
                    except json.JSONDecodeError:
                        await websocket.send_json({"error": "Invalid control message"})

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
