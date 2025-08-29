import whisper

model = whisper.load_model("large-v2")
## Used for testing New_Year_Resolution.mp3
result = model.transcribe(audio = "audios/New_Year_Resolution.mp3", language = "en",task="translate")
print(result["text"])