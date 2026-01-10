"""Fake data generator using Faker library for PII replacement."""

try:
    from faker import Faker

    HAS_FAKER = True
except ImportError:
    HAS_FAKER = False

from loclean.privacy.schemas import PIIEntity


class FakeDataGenerator:
    """Generator for fake PII data using Faker library."""

    def __init__(self, locale: str = "vi_VN") -> None:
        """
        Initialize fake data generator.

        Args:
            locale: Faker locale (e.g., "vi_VN", "en_US"). Defaults to "vi_VN".

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
