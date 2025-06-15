# ai/assistants/email_spam_classifier.py

from transformers import pipeline

class EmailSpamClassifier:
    def __init__(self, model_name="mrm8488/bert-tiny-finetuned-sms-spam-detection", device=-1):
        self.classifier = pipeline("text-classification", model=model_name, device=device)

    def classify(self, text):
        result = self.classifier(text)
        return {
            "label": result[0]['label'],
            "confidence": result[0]['score']
        }

# Example:
# esc = EmailSpamClassifier()
# print(esc.classify("Congratulations, you won a free iPhone! Click here."))
