import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = "", response_format: str = "text") -> str:
        """
        Generates completions given a prompt and system instructions.
        Response format can be 'text' or 'json'.
        """
        pass


class MockLLM(BaseLLM):
    """
    Offline Mock LLM that inspects ticket text for key phrases and generates
    realistic, structured JSON metadata and summaries.
    """
    def generate(self, prompt: str, system_instruction: str = "", response_format: str = "text") -> str:
        lower_prompt = prompt.lower()
        
        # Determine ticket context based on keywords
        if any(w in lower_prompt for w in ["refund", "billing", "charge", "payment", "invoice", "cost"]):
            intent = "Billing & Payments"
            product = "Payment Gateway / Billing Portal"
            sla = "High"
            sentiment = "Negative" if any(w in lower_prompt for w in ["wrong", "double", "error", "charged", "bad"]) else "Neutral"
            summary = "Customer is requesting a refund or inquiring about a charge on their account. They require transaction clarification and billing adjustment details."
            actions = [
                "Review the transaction log in Stripe/Billing Dashboard.",
                "Verify payment status and identify if a double-charge occurred.",
                "Issue a refund if the charge was unauthorized or duplicate.",
                "Send email confirmation of billing adjustment to user."
            ]
            kb = [
                {"title": "How to Process a Refund", "url": "https://kb.platform.com/billing/process-refund", "confidence": 0.95},
                {"title": "Billing Cycle and Charge Issues", "url": "https://kb.platform.com/billing/charge-issues", "confidence": 0.88}
            ]
            steps = ["Check customer account ID", "Look up invoice #4023", "Initiate refund action in Stripe", "Reply to customer with confirmation"]
        elif any(w in lower_prompt for w in ["password", "login", "account", "access", "reset", "sign in"]):
            intent = "Account Access & Authentication"
            product = "Identity Provider / User Account"
            sla = "Medium"
            sentiment = "Neutral"
            summary = "Customer is unable to access their account or needs to reset their password. They require support in resetting credentials."
            actions = [
                "Send password reset link to user's registered email.",
                "Verify customer identity through standard verification questions.",
                "Unlock account if locked due to multiple failed login attempts."
            ]
            kb = [
                {"title": "Resetting Your Account Password", "url": "https://kb.platform.com/auth/password-reset", "confidence": 0.97},
                {"title": "Unlocking Locked Accounts", "url": "https://kb.platform.com/auth/account-unlock", "confidence": 0.91}
            ]
            steps = ["Look up email in IAM dashboard", "Click 'Reset password' button", "Provide standard verification over chat", "Confirm user logged in successfully"]
        elif any(w in lower_prompt for w in ["slow", "bug", "crash", "down", "error", "broken", "fail"]):
            intent = "Technical Troubleshooting"
            product = "Web Application Dashboard"
            sla = "High"
            sentiment = "Negative"
            summary = "Customer reports technical issues, application crashes, or slow performance on the dashboard. They require diagnostic troubleshooting."
            actions = [
                "Request console logs or error screenshots from customer.",
                "Check system status page for active outages.",
                "Escalate diagnostic report to engineering team."
            ]
            kb = [
                {"title": "Dashboard Troubleshooting Guide", "url": "https://kb.platform.com/tech/dashboard-fix", "confidence": 0.93},
                {"title": "System Status and Active Outages", "url": "https://kb.platform.com/tech/system-outages", "confidence": 0.85}
            ]
            steps = ["Ask customer for console logs", "Cross-reference logs with Datadog/Sentry", "Verify browser version is supported", "Escalate to Tier 2 Engineering if unresolved"]
        else:
            intent = "General Inquiry"
            product = "Core Platform Services"
            sla = "Low"
            sentiment = "Positive" if "thanks" in lower_prompt or "thank you" in lower_prompt else "Neutral"
            summary = "Customer is asking a general question about platform usage or submitting a feature suggestion."
            actions = [
                "Provide standard platform documentation link.",
                "Log feature request in Product Management board.",
                "Close ticket after confirming user has no further issues."
            ]
            kb = [
                {"title": "Platform Quick Start Guide", "url": "https://kb.platform.com/general/quick-start", "confidence": 0.89},
                {"title": "Submitting Feature Requests", "url": "https://kb.platform.com/general/features", "confidence": 0.76}
            ]
            steps = ["Acknowledge receipt of ticket", "Provide relevant help center article", "Close ticket upon customer confirmation"]

        # Parse what exactly the prompt is asking for
        if response_format == "json":
            # Determine if prompt wants extraction metadata or recommendation format
            if "intent" in lower_prompt or "sla_priority" in lower_prompt:
                # Extractor schema
                result = {
                    "intent": intent,
                    "products": [product],
                    "sla_priority": sla,
                    "sentiment": sentiment,
                    "entities": [
                        {"name": "Stripe" if "refund" in lower_prompt else "AuthService", "label": "ORGANIZATION"},
                        {"name": "Customer Support", "label": "ROLE"}
                    ]
                }
                return json.dumps(result)
            elif "action" in lower_prompt or "kb" in lower_prompt or "canned" in lower_prompt:
                # Recommender schema
                result = {
                    "recommended_actions": actions,
                    "kb_articles": kb,
                    "runbook": {
                        "steps": steps,
                        "estimated_time_mins": 10 if sla == "High" else 5
                    }
                }
                return json.dumps(result)
            else:
                # Generic fallback JSON
                return json.dumps({
                    "summary": summary,
                    "intent": intent,
                    "sla": sla,
                    "sentiment": sentiment
                })
        else:
            # Plain text response (for summary prompts)
            return summary


class OpenAILLM(BaseLLM):
    """
    OpenAI-managed LLM (e.g. gpt-4o-mini).
    """
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system_instruction: str = "", response_format: str = "text") -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def get_llm(is_mock: bool = True, api_key: str = "", model_name: str = "gpt-4o-mini") -> BaseLLM:
    """
    Factory to return configured LLM.
    """
    if not is_mock and api_key:
        return OpenAILLM(api_key=api_key, model_name=model_name)
    else:
        return MockLLM()
