# import whisper

# model = whisper.load_model("large-v2")
# ## Used for testing New_Year_Resolution.mp3
# # result = model.transcribe(audio = "audios/New_Year_Resolution.mp3", language = "en",task="translate")
# result1 = model.transcribe(audio = "audios/sample.mp3", language = "en",task="translate")
# print(result1["text"])


from openai import OpenAI

client = OpenAI()
audio_file= open("audios/sample.mp3", "rb")

transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe", 
    file=audio_file
)

print(transcription.text)