# ai/assistants/code_explainer.py

from transformers import pipeline

class CodeExplainer:
    def __init__(self, model_name="Salesforce/codet5-base", device=-1):
        self.generator = pipeline("text2text-generation", model=model_name, device=device)
    
    def explain(self, code: str, detailed: bool = False) -> str:
        prompt = f"Explain this code{' in detail' if detailed else ''}:\n{code}"
        result = self.generator(prompt, max_length=256)
        return result[0]['generated_text']

# Example:
# ce = CodeExplainer()
# print(ce.explain("def foo(x): return x+1", detailed=True))
