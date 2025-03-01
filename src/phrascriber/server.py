import asyncio
import pyaudio
import numpy as np
import socket
from faster_whisper import WhisperModel

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Whisper model settings
model = "medium"
device = "cuda"
compute_type = "auto"
cpu_threads = 0
audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

# Async queue for transferring audio chunks
audio_queue = asyncio.Queue()
phrase_queue = asyncio.Queue()

HOST = "0.0.0.0"
PORT = 6969


async def handle_client(reader, writer):
    """Handles an incoming client connection and processes audio data."""
    print("Client connected.")
    frames = []
    receiving_audio = False

    try:
        while True:
            data = await reader.read(CHUNK * 2)  # Read raw PCM data
            if not data:
                break

            if data == b"END":
                if frames:
                    await audio_queue.put(frames)
                    frames = []
                continue

            frames.append(np.frombuffer(data, dtype=np.int16))

    except asyncio.CancelledError:
        print("Client handling task cancelled.")
    finally:
        print("Client disconnected.")
        writer.close()
        await writer.wait_closed()

async def transcribe_audio():
    """ Processes audio chunks from the queue and transcribes them. """
    try:
        while True:
            frames = await audio_queue.get()  # Wait for audio data from the queue
            audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0  # Normalize

            # Transcribe using Whisper
            segments, _ = audio_model.transcribe(audio_data, language="en")
            text = "".join(segment.text for segment in segments)
            phrase_queue.put_nowait(text)

            audio_queue.task_done()  # Mark task as done
    except asyncio.CancelledError:
        print("Transcription task cancelled.")


async def handle_transcription():
    """ Placeholder function to handle the transcription. """
    try:
        while True:
            phrase = await phrase_queue.get()
            print(f"Handling phrase: {phrase}")
    except asyncio.CancelledError:
        print("Handling task cancelled.")


async def async_main():
    """ Main async function to run both tasks concurrently. """
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"Server started on {HOST}:{PORT}")
    try:
        await asyncio.gather(
            server.serve_forever(),
            transcribe_audio(),
            handle_transcription(),
        )
    except asyncio.CancelledError:
        print("Main task cancelled.")

def main():
    """ Main function to run the async main function. """
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
