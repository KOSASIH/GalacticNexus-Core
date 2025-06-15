# ai/assistants/code_reviewer.py

import logging
from typing import List, Dict, Union, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CodeReviewer")

class CodeReviewer:
    SUPPORTED_MODELS = {
        "python": "Salesforce/codet5-base",
        "multi": "bigcode/starcoderbase",
        "secure": "microsoft/codebert-base"
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        review_type: str = "general"  # "general", "security", "performance"
    ):
        self.device = device
        self.review_type = review_type

        if not model_name:
            if review_type == "security":
                self.model_name = self.SUPPORTED_MODELS["secure"]
            elif review_type == "general":
                self.model_name = self.SUPPORTED_MODELS["python"]
            else:
                self.model_name = self.SUPPORTED_MODELS["multi"]
        else:
            self.model_name = model_name

        self.generator = pipeline("text2text-generation", model=self.model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        logger.info(f"Loaded code review model: {self.model_name}")

    def _build_prompt(self, code: str) -> str:
        if self.review_type == "security":
            prompt = (
                "Review the following code for security vulnerabilities and suggest improvements:\n"
                f"{code}\n"
            )
        elif self.review_type == "performance":
            prompt = (
                "Review the following code for performance issues and suggest optimizations:\n"
                f"{code}\n"
            )
        elif self.review_type == "refactor":
            prompt = (
                "Refactor the following code to improve maintainability and readability:\n"
                f"{code}\n"
            )
        else:
            prompt = (
                "Review the following code and provide suggestions for improvement, "
                "including correctness, best practices, style, security, and performance:\n"
                f"{code}\n"
            )
        return prompt

    def review(
        self,
        code_snippet: str,
        max_length: int = 256,
        explain: bool = False,
        return_metadata: bool = False
    ) -> Union[str, Dict]:
        """
        Reviews a single code snippet and provides suggestions.
        - explain: If True, returns reasoning with the review.
        - return_metadata: If True, returns metadata with the review.
        """
        prompt = self._build_prompt(code_snippet)
        result = self.generator(prompt, max_length=max_length)
        review_text = result[0]['generated_text']
        confidence = min(1.0, len(review_text) / max(1, len(code_snippet)))

        if explain or return_metadata:
            metadata = {
                "model": self.model_name,
                "review_type": self.review_type,
                "input_length": len(code_snippet),
                "review_length": len(review_text),
                "confidence": confidence
            }
            explanation = (
                f"Review generated using {self.model_name} for {self.review_type} review. "
                f"Confidence based on review/input length ratio: {confidence:.2f}"
            )
            result = {
                "review": review_text,
                "metadata": metadata,
                "explanation": explanation if explain else None
            }
            return result
        return review_text

    def batch_review(
        self,
        code_snippets: List[str],
        **kwargs
    ) -> List[Union[str, Dict]]:
        """Batch review for a list of code snippets."""
        return [self.review(snippet, **kwargs) for snippet in code_snippets]

# Example usage:
# cr = CodeReviewer(review_type="security")
# print(cr.review("def foo(x): return x+1", explain=True))
# cr = CodeReviewer(review_type="performance")
# print(cr.review("for i in range(1000000): print(i)"))
# print(cr.batch_review(["def add(a,b): return a+b", "while True: pass"], return_metadata=True))
