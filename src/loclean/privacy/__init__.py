"""Privacy-focused PII detection and scrubbing module.

This module provides functionality for detecting and masking/generating fake data
for Personally Identifiable Information (PII) such as names, phone numbers,
emails, credit cards, and addresses.
"""

from loclean.privacy.scrub import scrub_dataframe, scrub_string

__all__ = ["scrub_dataframe", "scrub_string"]

