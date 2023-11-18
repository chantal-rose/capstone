import re
import string

import prompts
from llm_utils import query_llm
from utils import tokenize

import wikipedia as wiki
from rank_bm25 import BM25Okapi


def get_topic(query: str, options: list[str] = []):
    if options:
        system_prompt = prompts.GET_TOPIC_WITH_OPTIONS
        user_prompt = f"options: {options}\nquery: {query}"
    else:
        system_prompt = prompts.GET_TOPIC_NO_OPTIONS
        user_prompt = f"query: {query}"
    topic = query_llm(messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ])
    return topic


def get_wiki_page(query: str, max_tries: int = 3) -> wiki.WikipediaPage:
    options = []
    for _ in range(max_tries):
        topic = get_topic(query, options)
        try:
            page = wiki.page(topic, auto_suggest=False)
        except wiki.exceptions.DisambiguationError as e:
            options = e.options
        except wiki.exceptions.PageError:
            options = wiki.search(topic)
        else:
            return page
    else:
        raise RuntimeError(f"No good topics found in {max_tries} attempts")


def get_n_best_paragraphs(page: wiki.WikipediaPage, query: str, n: int = 1) -> str:
    paragraphs = [text for t in re.split("=+ [^=]* =+", page.content) if (text := t.strip())]
    bm25 = BM25Okapi([tokenize(p) for p in paragraphs])
    return " ".join(bm25.get_top_n(tokenize(query), paragraphs, n))


def get_context(query: str) -> str:
    page = get_wiki_page(query)
    return get_n_best_paragraphs(page, query, n=1)


if __name__ == "__main__":
    while True:
        print(get_context(input()))
