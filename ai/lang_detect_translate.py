# ai/assistants/lang_detect_translate.py

from transformers import pipeline

class LangDetectTranslate:
    def __init__(self, translation_model="Helsinki-NLP/opus-mt-mul-en", device=-1):
        self.translator = pipeline("translation", model=translation_model, device=device)
        self.langid = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=device)

    def detect_language(self, text: str) -> dict:
        result = self.langid(text)
        return {"language": result[0]['label'], "confidence": result[0]['score']}

    def translate(self, text: str, target_lang: str = "en") -> str:
        return self.translator(text, tgt_lang=target_lang)[0]['translation_text']

    def detect_and_translate(self, text: str, target_lang: str = "en") -> dict:
        lang_info = self.detect_language(text)
        translation = self.translate(text, target_lang)
        return {
            "original_language": lang_info['language'],
            "confidence": lang_info['confidence'],
            "translated_text": translation
        }

# Example:
# ldt = LangDetectTranslate()
# print(ldt.detect_and_translate("Bonjour tout le monde!", "en"))
