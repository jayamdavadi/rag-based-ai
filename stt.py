## WHISPER USAGE
# import whisper

# model = whisper.load_model("large-v2")
# ## Used for testing New_Year_Resolution.mp3
# # result = model.transcribe(audio = "audios/New_Year_Resolution.mp3", language = "en",task="translate")
# result1 = model.transcribe(audio = "audios/sample.mp3", language = "en",task="translate",word_timestamps=False)
# print(result1["text"])

### Text to speech using OpenAI
from openai import OpenAI
from pydub import AudioSegment
import os
import json

# Initialize OpenAI client
client = OpenAI()  

# Load your audio
audio_path = "audios/sample.mp3"
audio = AudioSegment.from_file(audio_path)
audio_length_ms = len(audio)

# Define chunk duration in ms (3 seconds)
chunk_duration_ms = 3 * 1000  

# List to store chunks
chunks = []
current_time_ms = 0
chunk_index = 0

while current_time_ms < audio_length_ms:
    end_time_ms = min(current_time_ms + chunk_duration_ms, audio_length_ms)
    chunk_audio = audio[current_time_ms:end_time_ms]
    
    # Export temporary chunk
    chunk_filename = f"temp_chunk_{chunk_index}.mp3"
    chunk_audio.export(chunk_filename, format="mp3")
    
    # Transcribe using OpenAI
    with open(chunk_filename, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    
    # Append transcript with timestamps
    chunks.append({
        "start_time": current_time_ms / 1000,  # seconds
        "end_time": end_time_ms / 1000,
        "text": result.text
    })
    
    # Clean up temporary file
    os.remove(chunk_filename)
    
    # Move to next chunk
    current_time_ms += chunk_duration_ms
    chunk_index += 1

# Save to JSON file
output_data = {"chunks": chunks}
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("Transcription saved to output.json")
