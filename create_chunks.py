from openai import OpenAI
from pydub import AudioSegment
import os
import json

client = OpenAI()

# Ensure output folder exists
os.makedirs("json", exist_ok=True)

audio_folder = "audios"

# Loop through all mp3 files in the folder
for audio_file in os.listdir(audio_folder):
    if audio_file.lower().endswith(".mp3"):
        audio_path = os.path.join(audio_folder, audio_file)
        audio = AudioSegment.from_file(audio_path)
        audio_length_ms = len(audio)

        chunk_duration_ms = 10 * 1000  # 3 seconds
        chunks = []
        full_text_list = []
        current_time_ms = 0
        chunk_index = 0

        print(f"Processing {audio_file}...")

        while current_time_ms < audio_length_ms:
            end_time_ms = min(current_time_ms + chunk_duration_ms, audio_length_ms)
            chunk_audio = audio[current_time_ms:end_time_ms]

            chunk_filename = f"temp_chunk_{chunk_index}.mp3"
            chunk_audio.export(chunk_filename, format="mp3")

            with open(chunk_filename, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f
                )

            original_text = result.text.strip()

            # Translate only if text contains non-English characters
            if any(ord(c) > 127 for c in original_text):
                translation_result = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Translate this text to English."},
                        {"role": "user", "content": original_text}
                    ]
                )
                text = translation_result.choices[0].message.content.strip()
            else:
                text = original_text  # keep English text as-is

            chunks.append({
                "start_time": current_time_ms / 1000,
                "end_time": end_time_ms / 1000,
                "text": text
            })

            full_text_list.append(text)

            os.remove(chunk_filename)
            current_time_ms += chunk_duration_ms
            chunk_index += 1

        # Combine all chunk texts into full_text
        full_text = " ".join([t for t in full_text_list if t])

        output_data = {
            "chunks": chunks,
            "full_text": full_text
        }

        # Save JSON with same name as audio in json/ folder
        output_json_path = os.path.join("json", f"{os.path.splitext(audio_file)[0]}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"Saved transcription to {output_json_path}")

print("All files processed successfully!")
