"""Tests for fake data generator."""

import pytest

from loclean.privacy.generator import FakeDataGenerator
from loclean.privacy.schemas import PIIEntity


class TestFakeDataGenerator:
    """Test cases for FakeDataGenerator."""

    def test_generator_requires_faker(self) -> None:
        """Test that generator requires faker library."""
        try:
            from faker import Faker  # noqa: F401

            # Faker is installed, should work
            generator = FakeDataGenerator(locale="vi_VN")
            assert generator is not None
        except ImportError:
            # Faker not installed, should raise error
            with pytest.raises(ImportError):
                FakeDataGenerator(locale="vi_VN")

    def test_generate_fake_phone(self) -> None:
        """Test generating fake phone number."""
        try:
            from faker import Faker  # noqa: F401

            generator = FakeDataGenerator(locale="vi_VN")
            entity = PIIEntity(type="phone", value="0909123456", start=0, end=10)
            fake = generator.generate_fake(entity)

            assert fake is not None
            assert fake != entity.value
        except ImportError:
            pytest.skip("faker not installed")

    def test_generate_fake_email(self) -> None:
        """Test generating fake email."""
        try:
            from faker import Faker  # noqa: F401

            generator = FakeDataGenerator(locale="vi_VN")
            entity = PIIEntity(type="email", value="test@example.com", start=0, end=16)
            fake = generator.generate_fake(entity)

            assert fake is not None
            assert "@" in fake
            assert fake != entity.value
        except ImportError:
            pytest.skip("faker not installed")

    def test_generate_fake_person(self) -> None:
        """Test generating fake person name."""
        try:
            from faker import Faker  # noqa: F401

            generator = FakeDataGenerator(locale="vi_VN")
            entity = PIIEntity(type="person", value="Nguyễn Văn A", start=0, end=12)
            fake = generator.generate_fake(entity)

            assert fake is not None
            assert fake != entity.value
        except ImportError:
            pytest.skip("faker not installed")

    def test_generate_fake_unknown_type(self) -> None:
        """Test fallback for unknown entity type."""
        try:
            from faker import Faker  # noqa: F401

            generator = FakeDataGenerator(locale="vi_VN")
            # Use a type that doesn't have a specific generator
            entity = PIIEntity(type="ip_address", value="192.168.1.1", start=0, end=11)
            fake = generator.generate_fake(entity)

            # Should return something (either IP or fallback)
            assert fake is not None
        except ImportError:
            pytest.skip("faker not installed")
