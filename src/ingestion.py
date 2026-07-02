import re
from typing import List, Dict, Any

# Pre-compiled regular expressions for PII detection
EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
# Standard international / domestic phone regex
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
# Social Security Number (SSN) regex
SSN_REGEX = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
# Basic 16-digit credit card pattern
CREDIT_CARD_REGEX = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

def clean_text(text: str) -> str:
    """
    Cleans raw ticket text by removing extraneous whitespace and standardizing spacing.
    """
    if not text:
        return ""
    # Standardize whitespace characters
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

def parse_speaker_segments(transcript: str) -> List[Dict[str, str]]:
    """
    Parses dialogue text containing speaker labels (e.g. 'Customer: ...', 'Agent: ...', 'User: ...')
    into a structured list of speaker-text segments.
    """
    segments = []
    lines = transcript.split('\n')
    
    # Matches patterns like "Customer:", "Agent:", "Support:", "Client:", "User:"
    speaker_pattern = re.compile(r'^(Customer|Agent|Support|Client|User|Speaker\s+\d+):\s*(.*)$', re.IGNORECASE)
    
    current_speaker = "Unknown"
    current_text = []

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        match = speaker_pattern.match(line_strip)
        if match:
            # Save the previous segment if there is one
            if current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": clean_text(" ".join(current_text))
                })
            current_speaker = match.group(1).capitalize()
            current_text = [match.group(2)]
        else:
            if current_text:
                current_text.append(line_strip)
            else:
                # No speaker identified yet, initialize as Unknown
                current_speaker = "Unknown"
                current_text = [line_strip]
                
    if current_text:
        segments.append({
            "speaker": current_speaker,
            "text": clean_text(" ".join(current_text))
        })
        
    return segments

def mask_pii(text: str) -> str:
    """
    Masks PII (emails, phone numbers, SSNs, credit card numbers) from raw text.
    """
    if not text:
        return ""
    
    masked = text
    masked = EMAIL_REGEX.sub("[EMAIL]", masked)
    masked = PHONE_REGEX.sub("[PHONE]", masked)
    masked = SSN_REGEX.sub("[SSN]", masked)
    masked = CREDIT_CARD_REGEX.sub("[CREDIT_CARD]", masked)
    
    return masked

def preprocess_ticket(raw_text: str) -> Dict[str, Any]:
    """
    Full pipeline to clean, split speakers (if transcript), and mask PII.
    """
    cleaned = clean_text(raw_text)
    masked = mask_pii(cleaned)
    segments = parse_speaker_segments(raw_text)
    
    # Mask PII in individual segments as well
    for seg in segments:
        seg["text"] = mask_pii(seg["text"])
        
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "masked_text": masked,
        "segments": segments
    }
