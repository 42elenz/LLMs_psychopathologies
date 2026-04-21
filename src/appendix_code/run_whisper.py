import whisper
import os

# Path to the audio file
audio_path = "/zi/home/esra.lenz/Documents/00_HITKIP/01_GPTS/03_normative/00_DAIC-WOZ/300_AUDIO.wav"

# Load whisper large model (will download on first run ~2.9 GB)
print("Loading Whisper large model...")
model = whisper.load_model("large")

# Transcribe
print(f"Transcribing: {audio_path}")
result = model.transcribe(audio_path, language="en")

# Print full transcript
print("\n=== Transcript ===")
print(result["text"])

# Print segments with timestamps
print("\n=== Segments with timestamps ===")
for seg in result["segments"]:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"]
    print(f"[{start:7.2f}s - {end:7.2f}s] {text}")

# Save transcript to file
out_path = os.path.join(os.path.dirname(__file__), "whisper_transcript.txt")
with open(out_path, "w") as f:
    f.write(result["text"] + "\n\n")
    for seg in result["segments"]:
        f.write(f"[{seg['start']:7.2f}s - {seg['end']:7.2f}s] {seg['text']}\n")
print(f"\nTranscript saved to: {out_path}")
