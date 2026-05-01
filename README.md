# News Summarizer

A command-line news summarizer that fetches recent articles, summarizes them with OpenAI, analyzes sentiment with Anthropic Claude, and prints a cost summary for the run.

The app supports both normal processing and concurrent processing. It uses OpenAI for summaries, Anthropic Claude Sonnet 4.6 for sentiment analysis, and Claude Haiku 4.5 as an Anthropic fallback model.

## Quick Start

### 1. Install Python

Use Python 3.10 or newer.

Check your installation:

```powershell
python --version
```

If Windows opens the Microsoft Store or says Python was not found, install Python from python.org or disable the Microsoft Store Python aliases in Windows settings.

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file from the provided template:

```powershell
copy .env.example .env
```

Then edit `.env` and add your real keys:

```env
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
NEWS_API_KEY=your-eventregistry-api-key
```

Do not commit `.env`. It contains secrets.

### 4. Run the App

```powershell
python main.py
```

The app will ask for:

- news category: `technology`, `business`, `health`, or `general`
- number of articles: `1-10`
- whether to use async processing: `y` or `n`

## Example Output

```text
NEWS SUMMARIZER - Multi-Provider Edition

Enter news category (technology/business/health/general): technology
How many articles to process? (1-10): 3
Use async processing? (y/n): y

Fetching 3 articles from category: technology
Processing 3 articles concurrently...

NEWS SUMMARY REPORT
================================================================================

1. Example Technology Headline
   Source: Example News | Published: 2026-05-01
   URL: https://example.com/article

   SUMMARY:
   The article explains the main development in two or three sentences.

   SENTIMENT:
   Overall sentiment is neutral with high confidence. The tone is informative.

================================================================================
COST SUMMARY
================================================================================
Total requests: 6
Total cost: $0.0021
Total tokens: 3,842
  Input: 2,911
  Output: 931
Average cost per request: $0.000350
================================================================================

OK Processing complete!
```

Your exact output depends on current news results and model responses.

## Cost Notes

The app tracks estimated token costs for each LLM request.

Configured models:

- OpenAI summary model: `gpt-4o-mini`
- Anthropic primary model: `claude-sonnet-4-6`
- Anthropic fallback model: `claude-haiku-4-5`

Default daily budget:

```env
DAILY_BUDGET=5.00
```

If the estimated run cost exceeds the configured budget, the app raises a budget error. To reduce cost, process fewer articles or use async only when you need faster processing.

## File Map

```text
.
|-- .env.example          Example environment variables without real secrets
|-- .gitignore            Prevents local secrets and generated files from being committed
|-- README.md             User setup and usage guide
|-- config.py             Loads environment variables and central model settings
|-- llm_providers.py      OpenAI and Anthropic clients, fallback logic, cost tracking
|-- main.py               Interactive command-line entry point
|-- news_api.py           Fetches news articles from EventRegistry
|-- requirements.txt      Python package dependencies
|-- summarizer.py         Core summary, sentiment, report, and async processing logic
|-- test_summarizer.py    Unit tests for API parsing, providers, costs, and summarizer behavior
```

## Detailed Usage And Troubleshooting

### API Keys

You need three API keys:

- OpenAI API key for summaries
- Anthropic API key for sentiment analysis and fallback summaries
- EventRegistry API key for news articles

The code reads these from `.env` through `config.py`. If a key is missing, startup validation raises a clear error.

### Sync vs Async Mode

Use sync mode when you want simpler, sequential processing.

Use async mode when processing multiple articles. The async version runs article processing concurrently with `asyncio.to_thread`, while the provider SDK calls remain synchronous.

### Supported Categories

The command-line app accepts:

- `technology`
- `business`
- `health`
- `general`

If you enter another category, the app falls back to `technology`.

### Common Problems

`Python wurde nicht gefunden` or Microsoft Store opens:

Install Python 3.10+ and make sure `python` is available in your terminal. On Windows, disable App Execution Aliases for `python.exe` and `python3.exe` if they point to the Microsoft Store.

`Missing required configuration`:

Check that `.env` exists and contains `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `NEWS_API_KEY`.

No articles fetched:

Check your EventRegistry key, try another category, or reduce filters. The app returns an empty list when the news API request fails.

Budget exceeded:

Lower the number of articles or raise `DAILY_BUDGET` in `.env`.

Anthropic model unavailable:

The provider layer first tries `claude-sonnet-4-6`, then falls back to `claude-haiku-4-5`. If both are unavailable, it falls back to OpenAI.

### Running Tests

```powershell
python -m pytest test_summarizer.py -v
```

The tests use mocks for external services, so they should not call real APIs.
