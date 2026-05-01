"""News API integration module."""
import time

import requests

from config import Config


class NewsAPI:
    """Fetch news articles from EventRegistry."""

    def __init__(self):
        self.api_key = Config.NEWS_API_KEY
        self.base_url = "https://eventregistry.org/api/v1"
        self.last_call_time = 0
        self.min_interval = 60.0 / Config.NEWS_API_RPM

    def _wait_if_needed(self):
        """Wait if we need to rate limit."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"Rate limiting News API: waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
        self.last_call_time = time.time()

    def fetch_top_headlines(self, category="technology", country="us", max_articles=5):
        """
        Fetch top headlines.

        Args:
            category: News category or keyword.
            country: Country code (us, gb, de, fr, es).
            max_articles: Maximum number of articles to return.

        Returns:
            List of article dictionaries.
        """
        self._wait_if_needed()

        url = f"{self.base_url}/article/getArticles"
        country_locations = {
            "us": "http://en.wikipedia.org/wiki/United_States",
            "gb": "http://en.wikipedia.org/wiki/United_Kingdom",
            "de": "http://en.wikipedia.org/wiki/Germany",
            "fr": "http://en.wikipedia.org/wiki/France",
            "es": "http://en.wikipedia.org/wiki/Spain",
        }
        params = {
            "resultType": "articles",
            "apiKey": self.api_key,
            "keyword": category,
            "lang": "eng",
            "articlesSortBy": "date",
            "page": 1,
            "pageSize": max_articles,
            "articlesCount": max_articles,
        }

        if country and country.lower() in country_locations:
            params["sourceLocationUri"] = country_locations[country.lower()]

        try:
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", {}).get("results", [])
            if not isinstance(articles, list):
                raise ValueError("Unexpected news API response format")

            processed_articles = []
            for article in articles[:max_articles]:
                source_info = article.get("source", {})
                source_name = source_info.get("title") or source_info.get("uri") or "Unknown"
                body = article.get("body", "")
                published_at = (
                    article.get("dateTimePub")
                    or article.get("dateTime")
                    or article.get("date")
                )

                processed_articles.append(
                    {
                        "title": article.get("title", ""),
                        "description": body[:200],
                        "content": body,
                        "url": article.get("url", ""),
                        "source": source_name,
                        "published_at": published_at,
                    }
                )

            print(f"OK Fetched {len(processed_articles)} articles from News API")
            return processed_articles

        except requests.exceptions.RequestException as e:
            print(f"ERROR Error fetching news: {e}")
            return []
        except ValueError as e:
            print(f"ERROR Invalid news API response: {e}")
            return []


if __name__ == "__main__":
    api = NewsAPI()
    articles = api.fetch_top_headlines(category="technology", max_articles=3)

    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   URL: {article['url']}")
