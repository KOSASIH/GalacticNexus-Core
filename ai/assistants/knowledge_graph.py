# ai/assistants/knowledge_graph.py

import logging
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KnowledgeGraphBuilder")

class KnowledgeGraphBuilder:
    SUPPORTED_MODELS = {
        "en": "dslim/bert-base-NER",
        "multi": "Davlan/xlm-roberta-base-ner-hrl",
        "relations": "Babelscape/wikineural-multilingual-ner"
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: int = -1,
        aggregation_strategy: str = "simple"
    ):
        self.model_name = model_name or self.SUPPORTED_MODELS["en"]
        self.ner = pipeline("ner", model=self.model_name, aggregation_strategy=aggregation_strategy, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        logger.info(f"KnowledgeGraphBuilder initialized with model: {self.model_name}")

    def extract_entities(
        self,
        text: str,
        explain: bool = False,
        return_metadata: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extracts named entities from text.
        - explain: If True, includes model and confidence data.
        - return_metadata: If True, includes additional metadata.
        """
        results = self.ner(text)
        entities = []
        for ent in results:
            entity = {
                "entity": ent["entity_group"] if "entity_group" in ent else ent["entity"],
                "word": ent["word"],
                "score": ent["score"]
            }
            if explain or return_metadata:
                entity["model"] = self.model_name
            entities.append(entity)
        if return_metadata:
            return {
                "entities": entities,
                "meta": {
                    "model": self.model_name,
                    "input_length": len(text)
                }
            }
        return entities

    def extract_relationships(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Basic relationship extraction using pattern heuristics.
        For production, integrate with a true relation extraction model.
        """
        import re
        # Simple subject-verb-object pattern as a placeholder
        pattern = re.compile(r'([A-Z][a-zA-Z0-9_]+)\s+(was|is|created|developed|founded|by|in)\s+([A-Z][a-zA-Z0-9_]+|\d{4})')
        matches = pattern.findall(text)
        relationships = []
        for (subject, rel, obj) in matches:
            relationships.append({
                "subject": subject,
                "relation": rel,
                "object": obj
            })
        return relationships

    def build_knowledge_graph(
        self,
        text: str,
        explain: bool = False,
        return_metadata: bool = False
    ) -> Dict[str, Any]:
        """
        Builds a simple knowledge graph (entities and relationships) from text.
        """
        entities = self.extract_entities(text, explain=explain, return_metadata=False)
        relationships = self.extract_relationships(text)
        kg = {
            "entities": entities,
            "relationships": relationships
        }
        if return_metadata:
            kg["meta"] = {
                "model": self.model_name,
                "num_entities": len(entities),
                "num_relationships": len(relationships)
            }
        return kg

    def batch_extract(self, texts: List[str], **kwargs) -> List[Any]:
        """Batch entity extraction for multiple texts."""
        return [self.build_knowledge_graph(t, **kwargs) for t in texts]

# Example usage:
# kg = KnowledgeGraphBuilder()
# print(kg.extract_entities("GalacticNexus was created by KOSASIH in 2024.", explain=True))
# print(kg.build_knowledge_graph("GalacticNexus was created by KOSASIH in 2024 and maintained by the AI team.", return_metadata=True))
