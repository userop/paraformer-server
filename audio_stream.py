import asyncio
import json
import struct
import time
import websockets
from pydub import AudioSegment
import logging

# 关闭 websockets 调试日志（可选）
logging.getLogger("websockets").setLevel(logging.WARNING)

class AudioStreamingClient:
    def __init__(self, ws_uri: str, chunk_duration_sec: float = 1.0):
        self.ws_uri = ws_uri
        self.chunk_duration_sec = chunk_duration_sec
        self.websocket = None
        self.send_task = None
        self.recv_task = None

    async def connect(self):
        """建立 WebSocket 连接"""
        self.websocket = await websockets.connect(self.ws_uri)
        print("✅ WebSocket 连接已建立")

    async def send_audio(self, buffer):
        """协程：持续发送音频 PCM 块"""
        try:
            # 加载并标准化音频
            chunk_ms = 9600 * 2

            print("🔊 开始发送音频流...")
            for start in range(0, len(buffer), chunk_ms):
                end = start + chunk_ms
                pcm_data = buffer[start:end]
                b1 = struct.pack('>d', time.time())
                b2 = struct.pack('>I', len(pcm_data))
                # 8bit + 4bit
                await self.websocket.send(b1 + b2 + pcm_data)
                await asyncio.sleep(0.05)  # 避免发送过快（可选）
            print("⏹️ 音频发送完毕")
            await self.websocket.send(struct.pack('>d', time.time())+struct.pack('>I', 0))
        except Exception as e:
            print(f"发送错误: {e}")

    async def receive_results(self):
        """协程：持续接收服务端响应"""
        try:
            while 1:
                message = await self.websocket.recv()
                dom = json.loads(message)
                print(dom.get("text"))
                if dom["is_final"]:
                    break
                # 假设服务端返回 JSON 或文本
        except Exception as e:
            print(f"接收错误: {e}")

    async def run(self, data):
        """启动发送和接收协程"""
        await self.connect()
        # 并发运行两个任务
        self.send_task = asyncio.create_task(self.send_audio(data))
        self.recv_task = asyncio.create_task(self.receive_results())
        
        # 等待任一任务结束（或都结束）
        done, pending = await asyncio.wait(
            [self.send_task, self.recv_task],
            return_when=asyncio.ALL_COMPLETED
        )
        # 取消剩余任务
        for task in pending:
            task.cancel()

    async def close(self):
        if self.websocket:
            await self.websocket.close()


async def task(ws_uri: str, audio: bytes):
    client = AudioStreamingClient(ws_uri)
    try:
        await client.run(audio)
    finally:
        await client.close()


# 使用示例
async def main():
    ws_uri="ws://172.20.129.200:8400/ws/audio"
    audio_path = "/data1/resource/身份证.m4a"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    await task(ws_uri, audio.raw_data)


async def multi():
    ws_uri = "ws://172.20.129.200:8400/ws/audio"
    audio_path = "/data1/resource/身份证.m4a"
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    await asyncio.gather(*[task(ws_uri, audio.raw_data) for _ in range(5)])


if __name__ == "__main__":
    asyncio.run(main())