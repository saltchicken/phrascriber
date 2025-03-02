import asyncio
import pyaudio
import numpy as np

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0  # Seconds

class Client():
    def __init__(self, server_ip, server_port, receive_func=None):
        self.host = server_ip
        self.port = server_port
        if not receive_func:
            self.receive_func = lambda x: print(x)
        else:
            self.receive_func = receive_func

    async def send_audio(self, writer):
        """Captures audio from the microphone and sends speech data to the server."""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Listening for speech...")

        frames = []
        silent_chunks = 0
        recording = False

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)

                if np.max(np.abs(audio_np)) > SILENCE_THRESHOLD:
                    if not recording:
                        print("Speech detected. Recording...")
                        recording = True
                    frames.append(data)
                    silent_chunks = 0  # Reset silence counter
                elif recording:
                    silent_chunks += 1
                    if silent_chunks > int(SILENCE_DURATION * RATE / CHUNK):
                        print("Silence detected. Sending audio...")
                        if len(frames) > 7:  # Ensure sufficient audio length
                            for frame in frames:
                                writer.write(frame)
                                await writer.drain()

                            writer.write(b"END ")  # Send end marker
                            await writer.drain()
                        else:
                            print("Audio too short. Discarding.")

                        frames = []  # Reset buffer
                        recording = False

                await asyncio.sleep(0)  # Yield control to event loop
        except asyncio.CancelledError:
            print("Audio sending task cancelled.")
        finally:
            print("Closing audio stream.")
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def receive_transcriptions(self, reader):
        """Receives transcriptions from the server and prints them."""
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                transcription = data.decode().strip()
                self.receive_func(transcription)
        except asyncio.CancelledError:
            print("Receiving task cancelled.")

    async def main(self):
        """Handles both sending and receiving of data."""
        reader, writer = await asyncio.open_connection(self.host, self.port)
        print("Connected to server.")

        await asyncio.gather(self.send_audio(writer), self.receive_transcriptions(reader))

if __name__ == "__main__":
    receive_func = lambda x: print(f"Received: {x}")
    client = Client("10.0.0.2", "6969", receive_func)
    try:
        asyncio.run(client.main())
    except KeyboardInterrupt:
        print("Client shutting down.")

