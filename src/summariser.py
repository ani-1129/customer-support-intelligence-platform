from src.llm import BaseLLM

class TicketSummariser:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def summarise(self, ticket_text: str) -> str:
        """
        Generates a high-quality 2-3 sentence summary of the support ticket transcript.
        """
        system_instruction = (
            "You are an expert customer support supervisor. "
            "Your task is to write a concise, professional summary (2 to 3 sentences) of the customer support interaction. "
            "Highlight the customer's core issue, their sentiment, and the action taken or requested. "
            "Do not include greeting phrases, PII, or redundant pleasantries. Keep it focused and brief."
        )
        
        prompt = f"Support Ticket Content:\n\"\"\"\n{ticket_text}\n\"\"\"\n\nSummary:"
        
        summary = self.llm.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_format="text"
        )
        return summary.strip()
