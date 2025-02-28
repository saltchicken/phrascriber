import asyncio
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.0

# Whisper model settings
model = "medium"
device = "cuda"
compute_type = "auto"
cpu_threads = 0
audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

# Async queue for transferring audio chunks
audio_queue = asyncio.Queue()
phrase_queue = asyncio.Queue()


async def listen_to_microphone():
    """ Continuously listens to the microphone and pushes audio chunks into the queue. """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for speech...")

    frames = []
    silent_chunks = 0
    recording = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16)

        if np.max(np.abs(audio_np)) > SILENCE_THRESHOLD:
            if not recording:
                print("Speech detected. Recording...")
                recording = True
            frames.append(audio_np)
            silent_chunks = 0  # Reset silence counter
        elif recording:
            silent_chunks += 1
            if silent_chunks > int(SILENCE_DURATION * RATE / CHUNK):
                print("Silence detected. Stopping recording.")
                if frames:
                    if len(frames) <= 5:
                        print("Audio too short. Discarding.")
                    else:
                        await audio_queue.put(frames)  # Send audio data to processing task
                frames = []  # Reset frames buffer
                recording = False

        await asyncio.sleep(0)  # Yield control to avoid blocking

    stream.stop_stream()
    stream.close()
    p.terminate()


async def transcribe_audio():
    """ Processes audio chunks from the queue and transcribes them. """
    while True:
        frames = await audio_queue.get()  # Wait for audio data from the queue
        audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0  # Normalize

        # Transcribe using Whisper
        segments, _ = audio_model.transcribe(audio_data, language="en")
        text = "".join(segment.text for segment in segments)
        phrase_queue.put_nowait(text)

        audio_queue.task_done()  # Mark task as done

async def handle_transcription():
    """ Placeholder function to handle the transcription. """
    while True:
        phrase = await phrase_queue.get()
        print(f"Handling phrase: {phrase}")


async def main():
    """ Main async function to run both tasks concurrently. """
    handle_transcription_task = asyncio.create_task(handle_transcription())
    await asyncio.gather(
        listen_to_microphone(),
        transcribe_audio()
    )

# Run the event loop
asyncio.run(main())

