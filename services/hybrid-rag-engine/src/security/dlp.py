import re
from typing import Tuple, Dict

def luhn_checksum(card_number: str) -> bool:
    """Validates credit card number validity using Luhn mod-10 algorithm"""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0

class SensitiveDataRedactor:
    def __init__(self):
        self.ssn_regex = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        self.email_regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self.api_key_regex = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9_.\-]{25,})\b")
        self.phone_regex = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
        self.card_candidate_regex = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    def sanitize(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redacts PII and sensitive tokens from text, returning sanitized text and audit counts
        """
        if not text:
            return text, {}

        redaction_counts = {
            "credit_cards": 0,
            "ssn": 0,
            "api_keys": 0,
            "emails": 0,
            "phones": 0,
        }

        # 1. Redact API Keys / Bearer Secrets
        def replace_key(match):
            redaction_counts["api_keys"] += 1
            return "[REDACTED_API_KEY]"
        text = self.api_key_regex.sub(replace_key, text)

        # 2. Redact Social Security Numbers (SSN)
        def replace_ssn(match):
            redaction_counts["ssn"] += 1
            return "[REDACTED_SSN]"
        text = self.ssn_regex.sub(replace_ssn, text)

        # 3. Redact Validated Credit Cards with Luhn Check
        def replace_card(match):
            raw = match.group(0).replace(" ", "").replace("-", "")
            if luhn_checksum(raw):
                redaction_counts["credit_cards"] += 1
                return "[REDACTED_CREDIT_CARD]"
            return match.group(0)
        text = self.card_candidate_regex.sub(replace_card, text)

        # 4. Redact Emails
        def replace_email(match):
            redaction_counts["emails"] += 1
            return "[REDACTED_EMAIL]"
        text = self.email_regex.sub(replace_email, text)

        # 5. Redact Phone Numbers
        def replace_phone(match):
            redaction_counts["phones"] += 1
            return "[REDACTED_PHONE]"
        text = self.phone_regex.sub(replace_phone, text)

        # Remove keys with 0 count
        active_counts = {k: v for k, v in redaction_counts.items() if v > 0}
        return text, active_counts

redactor = SensitiveDataRedactor()
