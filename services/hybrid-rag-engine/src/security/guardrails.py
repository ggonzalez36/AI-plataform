import re
import base64
import logging
from typing import Tuple, Optional

logger = logging.getLogger("security.guardrails")

# Common Adversarial Prompt Injection Signatures
PROMPT_INJECTION_PATTERNS = [
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)\b", "INSTRUCTION_OVERRIDE", 0.95),
    (r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system)\s+(instructions|directives)\b", "INSTRUCTION_OVERRIDE", 0.95),
    (r"(?i)\b(show|reveal|display|output|print)\s+(your\s+)?(system\s+prompt|initial\s+instructions|instructions)\b", "SYSTEM_PROMPT_EXTRACTION", 0.90),
    (r"(?i)\byou\s+are\s+now\s+(DAN|unfiltered|unrestricted|in\s+developer\s+mode|evil\s+twin)\b", "ROLEPLAY_JAILBREAK", 0.95),
    (r"(?i)\bpretend\s+(you\s+can|you\s+have\s+no\s+rules|to\s+be\s+unbound)\b", "ROLEPLAY_JAILBREAK", 0.85),
    (r"(?i)(###\s*(System|Instruction|Assistant):|<\|im_start\|>|\[INST\]|---BEGIN SYSTEM PROMPT---)", "DELIMITER_HIJACKING", 0.90),
    (r"(?i)\bforget\s+all\s+(your\s+)?(training|guidelines|morals)\b", "INSTRUCTION_OVERRIDE", 0.90),
    (r"(?i)\bbypass\s+all\s+(safety|content)\s+filters\b", "FILTER_BYPASS", 0.95),
]

class PromptInjectionGuard:
    def __init__(self, block_threshold: float = 0.70):
        self.block_threshold = block_threshold

    def inspect(self, prompt: str) -> Tuple[bool, float, Optional[str]]:
        """
        Inspects query text for Prompt Injection and Jailbreaks.
        Returns: (is_blocked, threat_score, rule_violated)
        """
        if not prompt or len(prompt.strip()) == 0:
            return False, 0.0, None

        # 1. Direct Regex Pattern Matching
        for pattern, rule_name, score in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, prompt):
                logger.warning("🚨 [SECURITY_AUDIT] event=PROMPT_INJECTION_DETECTED rule=%s threat_score=%.2f snippet='%s'",
                               rule_name, score, prompt[:80])
                return True, score, rule_name

        # 2. Check for Base64 encoded injection evasion
        potential_b64_matches = re.findall(r"\b[A-Za-z0-9+/]{20,}={0,2}\b", prompt)
        for b64_str in potential_b64_matches:
            try:
                decoded = base64.b64decode(b64_str, validate=True).decode("utf-8", errors="ignore")
                for pattern, rule_name, score in PROMPT_INJECTION_PATTERNS:
                    if re.search(pattern, decoded):
                        logger.warning("🚨 [SECURITY_AUDIT] event=BASE64_INJECTION_DETECTED rule=%s threat_score=%.2f", rule_name, score)
                        return True, score, f"BASE64_ENCODED_{rule_name}"
            except Exception:
                continue

        return False, 0.0, None

guardrails = PromptInjectionGuard()
