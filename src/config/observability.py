import os
import re

def get_langsmith_client():
    """
    Initialize and return a LangSmith Client configured with a RuleNodeProcessor
    to automatically mask PII in traces.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return None

    try:
        from langsmith import Client
        from langsmith.anonymizer import RuleNodeProcessor, StringNodeRule
    except ImportError:
        return None

    # Define simple rules to catch obvious PII
    rules = [
        # Catch standard email formats
        StringNodeRule(
            pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
            replacement="[EMAIL_REDACTED]"
        ),
        # Catch standard 16-digit credit card formats (with or without spaces/dashes)
        StringNodeRule(
            pattern=re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            replacement="[CC_REDACTED]"
        ),
        # Catch US Social Security Numbers
        StringNodeRule(
            pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            replacement="[SSN_REDACTED]"
        )
    ]

    # Create the processor with our rules
    processor = RuleNodeProcessor(rules=rules)

    # Return a client instance that automatically uses this processor
    return Client(anonymizer=processor)
