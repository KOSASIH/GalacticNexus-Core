# ai/assistants/image_captioner.py

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

# Example:
# ic = ImageCaptioner()
# print(ic.caption("path/to/image.jpg"))
