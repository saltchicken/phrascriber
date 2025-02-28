import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono
RATE = 16000  # Sample rate
CHUNK = 1024  # Buffer size
SILENCE_THRESHOLD = 500  # Amplitude threshold for detecting silence
SILENCE_DURATION = 1.0  # Time in seconds before stopping recording

model = "medium"
device = "cuda"
compute_type = "auto"
cpu_threads = 0

audio_model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("Listening for speech...")

frames = []
silent_chunks = 0
recording = False

while True:
    data = stream.read(CHUNK)
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
            break

stream.stop_stream()
stream.close()
p.terminate()

# Convert frames to a single NumPy array
audio_data = np.concatenate(frames, axis=0).astype(np.float32) / 32768.0  # Normalize to -1.0 to 1.0

# Transcribe the audio data
segments, info = audio_model.transcribe(audio_data, language="en")

# Collect the transcribed text
text = "".join(segment.text for segment in segments)

print("Transcription:", text)

