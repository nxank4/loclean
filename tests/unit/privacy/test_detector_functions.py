"""Test cases for detector utility functions."""

from loclean.privacy.detector import find_all_positions


class TestFindAllPositions:
    """Test cases for find_all_positions function."""

    def test_find_single_occurrence(self) -> None:
        """Test finding single occurrence."""
        positions = find_all_positions("Contact 0909123456", "0909123456")

        assert len(positions) == 1
        assert positions[0] == (8, 18)

    def test_find_multiple_occurrences(self) -> None:
        """Test finding multiple occurrences."""
        positions = find_all_positions("Call 0909123456 or 0909123456", "0909123456")

        assert len(positions) == 2
        assert positions[0] == (5, 15)
        assert positions[1] == (19, 29)

    def test_find_no_occurrences(self) -> None:
        """Test finding no occurrences."""
        positions = find_all_positions("Contact support", "0909123456")

        assert len(positions) == 0

    def test_find_overlapping_occurrences(self) -> None:
        """Test finding overlapping occurrences."""
        positions = find_all_positions("aaaa", "aa")

        assert len(positions) == 3
        assert positions[0] == (0, 2)
        assert positions[1] == (1, 3)
        assert positions[2] == (2, 4)

    def test_find_at_start(self) -> None:
        """Test finding occurrence at start of text."""
        positions = find_all_positions("0909123456 is my number", "0909123456")

        assert len(positions) == 1
        assert positions[0] == (0, 10)

    def test_find_at_end(self) -> None:
        """Test finding occurrence at end of text."""
        positions = find_all_positions("My number is 0909123456", "0909123456")

        assert len(positions) == 1
        # "My number is " is 13 chars, so position starts at 13
        assert positions[0] == (13, 23)

    def test_find_empty_string(self) -> None:
        """Test finding in empty string."""
        positions = find_all_positions("", "test")

        assert len(positions) == 0

    def test_find_empty_value(self) -> None:
        """Test finding empty value - returns all positions for empty string."""
        positions = find_all_positions("test text", "")

        # Empty string matches at every position (including end)
        # This is expected behavior of str.find()
        assert len(positions) > 0

    def test_find_case_sensitive(self) -> None:
        """Test finding is case sensitive."""
        positions = find_all_positions("Test TEST test", "test")

        assert len(positions) == 1
        assert positions[0] == (10, 14)

    def test_find_with_special_characters(self) -> None:
        """Test finding with special characters."""
        positions = find_all_positions("Email: test@example.com", "test@example.com")

        assert len(positions) == 1
        # "Email: " is 7 chars, "test@example.com" is 16 chars
        assert positions[0] == (7, 23)

    def test_find_consecutive_occurrences(self) -> None:
        """Test finding consecutive occurrences."""
        positions = find_all_positions("ababab", "ab")

        assert len(positions) == 3
        assert positions[0] == (0, 2)
        assert positions[1] == (2, 4)
        assert positions[2] == (4, 6)
