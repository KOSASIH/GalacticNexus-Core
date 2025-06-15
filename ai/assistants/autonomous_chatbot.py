# ai/assistants/autonomous_chatbot.py

import logging
from typing import List, Dict, Optional, Any, Union
from transformers import pipeline, Conversation, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutonomousChatbot")

class AutonomousChatbot:
    SUPPORTED_MODELS = {
        "en": "microsoft/DialoGPT-medium",
        "blenderbot": "facebook/blenderbot-400M-distill",
        "llama": "meta-llama/Llama-2-7b-chat-hf"
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        max_history: int = 10,
        personality: Optional[str] = None,
        style: Optional[str] = None
    ):
        """
        model_name: Which conversational model to use
        device: -1 for CPU, or CUDA device id
        max_history: Number of previous exchanges to track
        personality: Custom persona string to inject into conversation
        style: Stylistic preference, e.g. "friendly", "concise", etc.
        """
        self.model_name = model_name or self.SUPPORTED_MODELS["en"]
        if self.model_name == self.SUPPORTED_MODELS["blenderbot"]:
            self.chatbot = pipeline("conversational", model=self.model_name, device=device)
        else:
            self.chatbot = pipeline("conversational", model=self.model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.personality = personality
        self.style = style
        logger.info(f"AutonomousChatbot initialized with model: {self.model_name}")

    def _build_conversation(self, user_input: str) -> Conversation:
        """
        Builds a Conversation object with history and persona/style.
        """
        prompt = ""
        if self.personality:
            prompt += f"[Personality: {self.personality}] "
        if self.style:
            prompt += f"[Style: {self.style}] "
        prompt += user_input

        conv = Conversation(prompt)
        # Add up to max_history previous exchanges
        for h in self.history[-self.max_history:]:
            conv.add_user_input(h['user'])
            conv.mark_processed()
            conv.append_response(h['bot'])
        return conv

    def chat(
        self,
        user_input: str,
        explain: bool = False,
        return_metadata: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Engage in a contextual chat, returning either a string or rich dict with metadata.
        """
        conv = self._build_conversation(user_input)
        response = self.chatbot(conv)
        answer = response.generated_responses[-1]
        self.history.append({'user': user_input, 'bot': answer})

        # Optional: Confidence scoring (length-based proxy)
        confidence = min(1.0, len(answer) / max(1, len(user_input)))

        if explain or return_metadata:
            metadata = {
                "model": self.model_name,
                "history_used": len(self.history),
                "personality": self.personality,
                "style": self.style,
                "confidence": confidence
            }
            explanation = f"Response generated using {self.model_name} with persona '{self.personality}' and style '{self.style}'."
            result = {
                "reply": answer,
                "metadata": metadata,
                "explanation": explanation if explain else None
            }
            return result
        return answer

    def batch_chat(self, user_inputs: List[str], **kwargs) -> List[Any]:
        """
        Batch chat for a list of user inputs.
        """
        return [self.chat(u, **kwargs) for u in user_inputs]

    def clear_history(self):
        """Reset the conversation history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation history."""
        return self.history

# Example usage:
# bot = AutonomousChatbot(personality="Wise mentor", style="friendly")
# print(bot.chat("Hello!", explain=True))
# print(bot.chat("Can you help me with a Python bug?"))
# print(bot.get_history())
