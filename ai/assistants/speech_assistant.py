# ai/assistants/speech_assistant.py

import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf

class SpeechAssistant:
    def __init__(self, stt_model="openai/whisper-base", tts_model="facebook/fastspeech2-en-ljspeech", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stt = pipeline("automatic-speech-recognition", model=stt_model, device=self.device)
        self.tts = pipeline("text-to-speech", model=tts_model, device=self.device)

    def speech_to_text(self, audio_path: str) -> str:
        return self.stt(audio_path)["text"]

    def text_to_speech(self, text: str, out_path: str = "output.wav"):
        output = self.tts(text)
        sf.write(out_path, output["audio"], output["sampling_rate"])
        return out_path

# Example:
# sa = SpeechAssistant()
# print(sa.speech_to_text("audio_sample.wav"))
# print(sa.text_to_speech("Welcome to GalacticNexus!", "welcome.wav"))
