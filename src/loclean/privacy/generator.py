"""Fake data generator using Faker library for PII replacement."""

try:
    from faker import Faker

    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

from loclean.privacy.schemas import PIIEntity


class FakeDataGenerator:
    """Generator for fake PII data using Faker library."""

    def __init__(self, locale: str = "en_US") -> None:
        """
        Initialize fake data generator.

        Args:
            locale: Faker locale (e.g., "en_US", "vi_VN"). Defaults to "en_US".

        Raises:
            ImportError: If faker library is not installed
        """
        if not HAS_FAKER:
            raise ImportError(
                "faker library is required for fake data generation. "
                "Install it with: pip install loclean[privacy]"
            )
        self.faker = Faker(locale)

    def generate_fake(self, entity: PIIEntity) -> str:
        """
        Generate fake data for a PII entity.

        Args:
            entity: PII entity to generate fake data for

        Returns:
            Fake data string matching the entity type
        """
        if entity.type == "phone":
            # Try to preserve format of original phone number
            original = entity.value
            # Remove all non-digit characters to check length
            digits_only = "".join(c for c in original if c.isdigit())

            # Generate fake number based on original format
            if len(digits_only) <= 7:
                # Short format (e.g., 555-1234) - use simple US format
                fake_digits = self.faker.numerify("#######")
                # Try to preserve separator style
                if "-" in original:
                    return f"{fake_digits[:3]}-{fake_digits[3:]}"
                elif " " in original:
                    return f"{fake_digits[:3]} {fake_digits[3:]}"
                elif "." in original:
                    return f"{fake_digits[:3]}.{fake_digits[3:]}"
                else:
                    return fake_digits
            elif len(digits_only) <= 10:
                # Standard US format (e.g., 555-123-4567)
                fake_digits = self.faker.numerify("##########")
                if "(" in original and ")" in original:
                    return f"({fake_digits[:3]}) {fake_digits[3:6]}-{fake_digits[6:]}"
                elif "-" in original:
                    parts = original.split("-")
                    if len(parts) == 3:
                        return f"{fake_digits[:3]}-{fake_digits[3:6]}-{fake_digits[6:]}"
                    else:
                        return f"{fake_digits[:3]}-{fake_digits[3:]}"
                elif " " in original:
                    return f"{fake_digits[:3]} {fake_digits[3:6]} {fake_digits[6:]}"
                else:
                    return fake_digits
            else:
                # International or long format - use faker's phone_number
                return str(self.faker.phone_number())
        elif entity.type == "email":
            return str(self.faker.email())
        elif entity.type == "person":
            return str(self.faker.name())
        elif entity.type == "credit_card":
            return str(self.faker.credit_card_number())
        elif entity.type == "address":
            return str(self.faker.address())
        elif entity.type == "ip_address":
            # Randomly choose IPv4 or IPv6
            import random

            if random.random() < 0.5:
                return str(self.faker.ipv4())
            else:
                return str(self.faker.ipv6())
        else:
            # Fallback to mask format
            return f"[{entity.type.upper()}]"
