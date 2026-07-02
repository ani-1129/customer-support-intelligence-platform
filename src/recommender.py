import json
from typing import Dict, Any, List
from src.llm import BaseLLM

class AgentRecommender:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate_recommendations(self, ticket_text: str, historical_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates recommended next-actions, KB articles, and agent runbook steps
        based on the ticket and retrieved historical matches.
        """
        # Format historical tickets into context
        formatted_context = ""
        for i, doc in enumerate(historical_context):
            formatted_context += f"--- Past Ticket {i+1} ---\n"
            formatted_context += f"Content: {doc.get('text', '')}\n"
            metadata = doc.get("metadata", {})
            if metadata:
                formatted_context += f"Resolutions/Metadata: {json.dumps(metadata)}\n"
            formatted_context += "\n"

        system_instruction = (
            "You are an AI Agent Advisor. Based on the current customer support ticket and "
            "provided historical context of similar tickets, generate a JSON object containing:\n"
            "1. 'recommended_actions': A list of string descriptions representing concrete next-actions for the agent.\n"
            "2. 'kb_articles': A list of objects with keys 'title', 'url', and 'confidence' (float, 0-1) representing relevant KB resources.\n"
            "3. 'runbook': An object with keys:\n"
            "   - 'steps': A list of string steps to troubleshoot/resolve this specific issue in order.\n"
            "   - 'estimated_time_mins': Integer estimation of time needed to execute the runbook.\n\n"
            "Respond ONLY with valid JSON."
        )

        prompt = (
            f"Current Support Ticket:\n\"\"\"\n{ticket_text}\n\"\"\"\n\n"
            f"Historical Context:\n\"\"\"\n{formatted_context}\n\"\"\"\n\n"
            f"Recommendations JSON:"
        )

        response = self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_format="json"
        )

        clean_response = response.strip()
        if clean_response.startswith("```"):
            lines = clean_response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_response = "\n".join(lines).strip()

        try:
            recommendations = json.loads(clean_response)
        except json.JSONDecodeError:
            # Fallback values
            recommendations = {
                "recommended_actions": ["Acknowledge receipt and investigate details."],
                "kb_articles": [{"title": "General Troubleshooting Guide", "url": "https://kb.platform.com/general", "confidence": 0.5}],
                "runbook": {
                    "steps": ["Acknowledge ticket", "Contact customer support lead if complex"],
                    "estimated_time_mins": 5
                }
            }

        recommendations.setdefault("recommended_actions", ["Acknowledge receipt and investigate details."])
        recommendations.setdefault("kb_articles", [{"title": "General Troubleshooting Guide", "url": "https://kb.platform.com/general", "confidence": 0.5}])
        recommendations.setdefault("runbook", {
            "steps": ["Acknowledge ticket", "Review historical records"],
            "estimated_time_mins": 5
        })

        return recommendations
