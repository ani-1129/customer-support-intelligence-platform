import json
from typing import Dict, Any, List
from src.llm import BaseLLM

class MetadataExtractor:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def extract_metadata(self, ticket_text: str) -> Dict[str, Any]:
        """
        Extracts structured metadata (intent, products, SLA, sentiment, entities) from a ticket.
        """
        system_instruction = (
            "You are a structured data extractor. You must analyze the customer support interaction "
            "and output a JSON object containing exactly the following keys:\n"
            "- 'intent': A short, clear category of the customer's request (e.g., 'Billing Query', 'Password Reset', 'Bug Report').\n"
            "- 'products': A list of products or platform modules mentioned in the text.\n"
            "- 'sla_priority': One of: 'High', 'Medium', 'Low' based on urgency, frustration, and critical failure indicators.\n"
            "- 'sentiment': One of: 'Positive', 'Negative', 'Neutral'.\n"
            "- 'entities': A list of dictionary objects with keys 'name' and 'label' (e.g., product names, names of external tools, order IDs, account IDs).\n"
            "Respond ONLY with valid JSON. Do not include markdown code block formatting (like ```json ... ```) in your API output if possible, or ensure it is easily parseable."
        )

        prompt = f"Support Ticket Text:\n\"\"\"\n{ticket_text}\n\"\"\"\n\nExtracted JSON Metadata:"
        
        response = self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_format="json"
        )
        
        # Clean response if LLM enclosed it in markdown code blocks
        clean_response = response.strip()
        if clean_response.startswith("```"):
            # Strip code fences if present
            lines = clean_response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_response = "\n".join(lines).strip()

        try:
            metadata = json.loads(clean_response)
        except json.JSONDecodeError:
            # Simple fallback structure
            metadata = {
                "intent": "General Inquiry",
                "products": [],
                "sla_priority": "Low",
                "sentiment": "Neutral",
                "entities": []
            }
            
        # Ensure schema defaults if missing
        metadata.setdefault("intent", "General Inquiry")
        metadata.setdefault("products", [])
        metadata.setdefault("sla_priority", "Low")
        metadata.setdefault("sentiment", "Neutral")
        metadata.setdefault("entities", [])
        
        return metadata
