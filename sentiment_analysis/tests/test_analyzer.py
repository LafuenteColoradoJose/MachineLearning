"""Tests for the sentiment analyzer module."""

import pytest

from sentiment.analyzer import analyze_sentiment


class TestAnalyzeSentiment:
    """Test suite for analyze_sentiment function."""

    @pytest.mark.parametrize(
        "text, expected_sentiment, expected_score",
        [
            ("I love this!", "positivo", 0.8),
            ("I hate this.", "negativo", -0.8),
            ("This is amazing", "positivo", 0.6),
            ("This is terrible", "negativo", -0.6),
            ("Hello world", "positivo", 0.0),  # neutral default
        ],
    )
    def test_basic_sentiments(
        self,
        text: str,
        expected_sentiment: str,
        expected_score: float,
    ) -> None:
        """Test basic positive and negative sentiment detection."""
        result = analyze_sentiment(text)
        assert result["sentiment"] == expected_sentiment
        assert result["score"] == expected_score

    def test_empty_string(self) -> None:
        """Test that empty strings raise ValueError."""
        with pytest.raises(ValueError, match="Input text is empty or whitespace only"):
            analyze_sentiment("")

    def test_whitespace_only(self) -> None:
        """Test that whitespace-only strings raise ValueError."""
        with pytest.raises(ValueError, match="Input text is empty or whitespace only"):
            analyze_sentiment("   ")

    def test_none_input(self) -> None:
        """Test that None input raises TypeError."""
        with pytest.raises(TypeError, match="Expected string, got NoneType"):
            analyze_sentiment(None)  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        """Test that non-string inputs raise TypeError."""
        with pytest.raises(TypeError, match="Expected string, got int"):
            analyze_sentiment(123)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "text",
        [
            "AMOR",  # uppercase Spanish
            "exCELente",  # mixed case Spanish
        ],
    )
    def test_case_insensitivity(self, text: str) -> None:
        """Test that sentiment analysis is case-insensitive."""
        result = analyze_sentiment(text)
        # Both "AMOR" and "exCELente" should be detected as positive
        assert result["sentiment"] == "positivo"


# Nuevos tests para aumentar cobertura - casos borde
class TestAnalyzerEdgeCases:
    """Test edge cases for analyzer coverage."""

    def test_very_positive_text(self) -> None:
        """Test with many positive words to trigger score clipping at 1.0."""
        # "love" has weight 0.8, so multiple love words should clip
        text = "love love love love love"
        result = analyze_sentiment(text)
        # Score should be clipped to 1.0
        assert result["sentiment"] == "positivo"
        assert result["score"] == 1.0

    def test_very_negative_text(self) -> None:
        """Test with many negative words to trigger score clipping at -1.0."""
        text = "hate hate hate hate hate"
        result = analyze_sentiment(text)
        # Score should be clipped to -1.0
        assert result["sentiment"] == "negativo"
        assert result["score"] == -1.0

    def test_mixed_positive_negative(self) -> None:
        """Test with equal positive and negative words."""
        text = "love hate"
        result = analyze_sentiment(text)
        # Should default to positive with score 0.0 when balanced
        assert result["sentiment"] == "positivo"
        assert result["score"] == 0.0

    def test_single_positive_word(self) -> None:
        """Test single positive word detection."""
        result = analyze_sentiment("love")
        assert result["sentiment"] == "positivo"
        assert result["score"] > 0

    def test_single_negative_word(self) -> None:
        """Test single negative word detection."""
        result = analyze_sentiment("hate")
        assert result["sentiment"] == "negativo"
        assert result["score"] < 0

    def test_only_neutral_text(self) -> None:
        """Test text with no sentiment words."""
        result = analyze_sentiment("the cat sits on the mat")
        # Should default to positive with score 0.0
        assert result["sentiment"] == "positivo"
        assert result["score"] == 0.0
