"""Main application entry point."""
import asyncio
import sys

from summarizer import AsyncNewsSummarizer, NewsSummarizer


VALID_CATEGORIES = {"technology", "business", "health", "general"}


def get_category():
    """Prompt for a supported news category."""
    prompt = "\nEnter news category (technology/business/health/general): "
    category = input(prompt).strip().lower() or "technology"

    if category not in VALID_CATEGORIES:
        print(f"Unknown category '{category}'. Using technology.")
        return "technology"

    return category


def get_num_articles():
    """Prompt for the number of articles, clamped to 1-10."""
    raw_value = input("How many articles to process? (1-10): ").strip()

    try:
        num_articles = int(raw_value)
    except ValueError:
        print("Invalid article count. Using 3.")
        return 3

    return max(1, min(10, num_articles))


def get_use_async():
    """Prompt for sync or async processing."""
    return input("Use async processing? (y/n): ").strip().lower() == "y"


def run_sync(category, num_articles):
    """Run the synchronous summarizer workflow."""
    summarizer = NewsSummarizer()
    articles = summarizer.news_api.fetch_top_headlines(
        category=category,
        max_articles=num_articles,
    )

    if not articles:
        print("No articles fetched. Check your News API key or try another category.")
        return

    print(f"\nProcessing {len(articles)} articles...")
    results = summarizer.process_articles(articles)
    summarizer.generate_report(results)


def run_async(category, num_articles):
    """Run the asynchronous summarizer workflow."""
    summarizer = AsyncNewsSummarizer()
    articles = summarizer.news_api.fetch_top_headlines(
        category=category,
        max_articles=num_articles,
    )

    if not articles:
        print("No articles fetched. Check your News API key or try another category.")
        return

    print(f"\nProcessing {len(articles)} articles concurrently...")
    results = asyncio.run(summarizer.process_articles_async(articles, max_concurrent=3))
    summarizer.generate_report(results)


def main():
    """Run the news summarizer."""
    print("=" * 80)
    print("NEWS SUMMARIZER - Multi-Provider Edition")
    print("=" * 80)

    category = get_category()
    num_articles = get_num_articles()
    use_async = get_use_async()

    print(f"\nFetching {num_articles} articles from category: {category}")

    try:
        if use_async:
            run_async(category, num_articles)
        else:
            run_sync(category, num_articles)

        print("\nOK Processing complete!")

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\nERROR {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
