import logging
import os
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"), override=False)

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.4
GROQ_MAX_TOKENS = 1024
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REQUEST_TIMEOUT = 8

CHAT_SYSTEM_PROMPT = """You are an expert credit risk analyst and financial advisor AI assistant.

RULES:
1. Answer questions about credit risk, lending, loan analysis, and financial regulations.
2. If the user provides web articles as context, reference them in your answer where relevant.
3. Be concise but thorough. Use bullet points for clarity.
4. If you don't know something, say so — never fabricate facts.
5. When discussing regulations, mention that rules vary by jurisdiction.
6. Always remind users that your advice is informational, not financial advice.
7. If the question is unrelated to finance/credit, politely redirect.
"""


def fetch_financial_articles(query: str = "credit risk lending", max_articles: int = 5) -> List[Dict]:
    try:
        params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        response = requests.get(GOOGLE_NEWS_RSS, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        items = root.findall(".//item")

        articles = []
        for item in items[:max_articles]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            source = item.findtext("source", "")

            clean_title = title.rsplit(" - ", 1)[0] if " - " in title else title

            articles.append({
                "title": clean_title,
                "link": link,
                "source": source or (title.rsplit(" - ", 1)[-1] if " - " in title else ""),
                "published": pub_date,
            })

        logger.info("Fetched %d articles for query: '%s'", len(articles), query)
        return articles

    except Exception as e:
        logger.warning("Article fetch failed: %s", e)
        return []


def _format_articles_context(articles: List[Dict]) -> str:
    if not articles:
        return ""
    lines = ["\n--- RECENT FINANCIAL NEWS (for reference) ---"]
    for i, article in enumerate(articles, 1):
        lines.append(f"{i}. \"{article['title']}\" — {article['source']} ({article['published']})")
    lines.append("---")
    return "\n".join(lines)


def get_chat_response(user_message: str, articles: Optional[List[Dict]] = None, conversation_history: Optional[List[Dict]] = None) -> str:
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return (
            "💡 **LLM Chat requires a GROQ_API_KEY.**\n\n"
            "To enable the AI chat, add your Groq API key to your `.env` file:\n"
            "```\nGROQ_API_KEY=your_key_here\n```\n"
            "You can get a free key at [console.groq.com](https://console.groq.com)."
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)

        full_message = user_message
        if articles:
            full_message += _format_articles_context(articles)

        messages.append({"role": "user", "content": full_message})

        completion = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            stream=False,
        )

        response = completion.choices[0].message.content
        logger.info("Chat response generated (%d chars)", len(response))
        return response

    except Exception as e:
        logger.error("Chat response failed: %s", e)
        return f"⚠️ Error generating response: {e}"
