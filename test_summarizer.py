"""Unit tests for news summarizer."""
import asyncio
from unittest.mock import Mock, patch

import pytest

from news_api import NewsAPI
from llm_providers import LLMProviders, CostTracker, count_tokens
from summarizer import AsyncNewsSummarizer, NewsSummarizer


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_track_request(self):
        """Test tracking a single request."""
        tracker = CostTracker()
        cost = tracker.track_request("openai", "gpt-4o-mini", 100, 500)

        assert cost > 0
        assert tracker.total_cost == cost
        assert len(tracker.requests) == 1

    def test_get_summary(self):
        """Test summary generation."""
        tracker = CostTracker()
        tracker.track_request("openai", "gpt-4o-mini", 100, 200)
        tracker.track_request("anthropic", "claude-sonnet-4-6", 150, 300)

        summary = tracker.get_summary()

        assert summary["total_requests"] == 2
        assert summary["total_cost"] > 0
        assert summary["total_input_tokens"] == 250
        assert summary["total_output_tokens"] == 500

    def test_budget_check(self):
        """Test budget checking."""
        tracker = CostTracker()

        tracker.track_request("openai", "gpt-4o-mini", 100, 100)
        tracker.check_budget(10.00)

        tracker.total_cost = 15.00
        with pytest.raises(Exception, match="budget.*exceeded"):
            tracker.check_budget(10.00)


class TestTokenCounting:
    """Test token counting."""

    def test_count_tokens(self):
        """Test token counting function."""
        text = "Hello, how are you?"
        count = count_tokens(text)

        assert count > 0
        assert count < len(text)


class TestNewsAPI:
    """Test News API integration."""

    @patch("news_api.requests.get")
    def test_fetch_top_headlines(self, mock_get):
        """Test fetching headlines from the EventRegistry-style response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": {
                "results": [
                    {
                        "title": "Test Article",
                        "body": "Test content",
                        "url": "https://example.com",
                        "source": {"title": "Test Source"},
                        "dateTimePub": "2026-01-19",
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        api = NewsAPI()
        articles = api.fetch_top_headlines(max_articles=1)

        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["description"] == "Test content"
        assert articles[0]["content"] == "Test content"
        assert articles[0]["source"] == "Test Source"
        assert articles[0]["published_at"] == "2026-01-19"

        request_url = mock_get.call_args.args[0]
        request_params = mock_get.call_args.kwargs["params"]
        assert request_url.endswith("/article/getArticles")
        assert request_params["resultType"] == "articles"
        assert request_params["keyword"] == "technology"
        assert request_params["pageSize"] == 1

    @patch("news_api.requests.get")
    def test_fetch_top_headlines_rejects_unexpected_response(self, mock_get):
        """Test invalid response shapes are handled gracefully."""
        mock_response = Mock()
        mock_response.json.return_value = {"articles": {"results": {}}}
        mock_get.return_value = mock_response

        api = NewsAPI()
        articles = api.fetch_top_headlines(max_articles=1)

        assert articles == []


class TestLLMProviders:
    """Test LLM provider integration."""

    @patch("llm_providers.Anthropic")
    @patch("llm_providers.OpenAI")
    def test_ask_openai(self, mock_openai_class, mock_anthropic_class):
        """Test OpenAI integration."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        mock_anthropic_class.return_value = Mock()

        providers = LLMProviders()

        response = providers.ask_openai("Test prompt")

        assert response == "Test response"
        assert mock_client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"

    @patch("llm_providers.Anthropic")
    @patch("llm_providers.OpenAI")
    def test_ask_anthropic_tries_haiku_when_sonnet_404(
        self, mock_openai_class, mock_anthropic_class
    ):
        """Test Anthropic falls from Sonnet 4.6 to Haiku 4.5 on model 404."""
        mock_client = Mock()
        model_error = Exception("not_found_error: model: claude-sonnet-4-6")
        model_error.status_code = 404
        mock_response = Mock()
        mock_response.content = [Mock(text="Haiku response")]
        mock_client.messages.create.side_effect = [model_error, mock_response]
        mock_anthropic_class.return_value = mock_client
        mock_openai_class.return_value = Mock()

        providers = LLMProviders()
        response = providers.ask_anthropic("Test prompt")

        assert response == "Haiku response"
        assert mock_client.messages.create.call_count == 2
        assert (
            mock_client.messages.create.call_args_list[0].kwargs["model"]
            == "claude-sonnet-4-6"
        )
        assert (
            mock_client.messages.create.call_args_list[1].kwargs["model"]
            == "claude-haiku-4-5"
        )

    @patch("llm_providers.Anthropic")
    @patch("llm_providers.OpenAI")
    def test_ask_anthropic_falls_back_to_openai_when_all_models_404(
        self, mock_openai_class, mock_anthropic_class
    ):
        """Test OpenAI fallback when no Anthropic model is available."""
        mock_anthropic_client = Mock()
        model_error = Exception("not_found_error: model: unavailable")
        model_error.status_code = 404
        mock_anthropic_client.messages.create.side_effect = model_error
        mock_anthropic_class.return_value = mock_anthropic_client

        mock_openai_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="OpenAI fallback response"))]
        mock_openai_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_client

        providers = LLMProviders()
        response = providers.ask_anthropic("Test prompt")

        assert response == "OpenAI fallback response"
        assert mock_openai_client.chat.completions.create.called


class TestNewsSummarizer:
    """Test news summarizer."""

    @patch("summarizer.LLMProviders")
    @patch("summarizer.NewsAPI")
    def test_initialization(self, mock_news_api_class, mock_llm_providers_class):
        """Test summarizer initialization."""
        summarizer = NewsSummarizer()

        assert summarizer.news_api == mock_news_api_class.return_value
        assert summarizer.llm_providers == mock_llm_providers_class.return_value

    def test_summarize_article(self):
        """Test article summarization and sentiment analysis."""
        summarizer = NewsSummarizer.__new__(NewsSummarizer)
        summarizer.news_api = Mock()
        summarizer.llm_providers = Mock()
        summarizer.llm_providers.ask_openai.return_value = "Test summary"
        summarizer.llm_providers.ask_anthropic.return_value = "Positive sentiment"

        article = {
            "title": "Test Article",
            "description": "Test description",
            "content": "Test content",
            "url": "https://example.com",
            "source": "Test Source",
            "published_at": "2026-01-19",
        }

        result = summarizer.summarize_article(article)

        assert result["title"] == "Test Article"
        assert result["summary"] == "Test summary"
        assert result["sentiment"] == "Positive sentiment"
        summarizer.llm_providers.ask_openai.assert_called_once()
        summarizer.llm_providers.ask_anthropic.assert_called_once()

    def test_summarize_article_falls_back_to_anthropic_for_summary(self):
        """Test Anthropic generates the summary when OpenAI fails."""
        summarizer = NewsSummarizer.__new__(NewsSummarizer)
        summarizer.news_api = Mock()
        summarizer.llm_providers = Mock()
        summarizer.llm_providers.ask_openai.side_effect = Exception("OpenAI down")
        summarizer.llm_providers.ask_anthropic.side_effect = [
            "Fallback summary",
            "Neutral sentiment",
        ]

        article = {
            "title": "Fallback Article",
            "description": "Fallback description",
            "content": "Fallback content",
            "url": "https://example.com/fallback",
            "source": "Test Source",
            "published_at": "2026-01-19",
        }

        result = summarizer.summarize_article(article)

        assert result["summary"] == "Fallback summary"
        assert result["sentiment"] == "Neutral sentiment"
        assert summarizer.llm_providers.ask_anthropic.call_count == 2

    def test_process_articles_skips_failed_articles(self):
        """Test processing continues when one article fails."""
        summarizer = NewsSummarizer.__new__(NewsSummarizer)
        summarizer.summarize_article = Mock(
            side_effect=[
                {"title": "Good Article"},
                Exception("Bad article"),
                {"title": "Another Good Article"},
            ]
        )

        results = summarizer.process_articles([{}, {}, {}])

        assert results == [{"title": "Good Article"}, {"title": "Another Good Article"}]

    def test_process_articles_async_filters_failures(self):
        """Test async processing returns successful article results only."""
        summarizer = AsyncNewsSummarizer.__new__(AsyncNewsSummarizer)

        def summarize_article(article):
            if article["title"] == "Bad Article":
                raise Exception("Bad article")
            return {"title": article["title"]}

        summarizer.summarize_article = Mock(side_effect=summarize_article)

        articles = [
            {"title": "Good Article"},
            {"title": "Bad Article"},
            {"title": "Another Good Article"},
        ]
        results = asyncio.run(summarizer.process_articles_async(articles, max_concurrent=2))

        assert results == [{"title": "Good Article"}, {"title": "Another Good Article"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
