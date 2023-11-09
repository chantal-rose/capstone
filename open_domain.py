import re
import string

import wikipedia as wiki

from nltk import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


def get_topic(query: str, options: list[str] = None):
    # Mock function
    if options:
        return options[0]
    else:
        return query.rstrip("?").split(" ")[-1]


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
    def tokenize(text: str):
        stops = set(stopwords.words("english")) | set(string.punctuation)
        return [t for t in word_tokenize(text.lower()) if t not in stops]

    paragraphs = [text for t in re.split("=+ [^=]* =+", page.content) if (text := t.strip())]
    bm25 = BM25Okapi([tokenize(p) for p in paragraphs])
    return " ".join(bm25.get_top_n(tokenize(query), paragraphs, n))


def get_content(query: str) -> str:
    page = get_wiki_page(query)
    return get_n_best_paragraphs(page, query, n=1)


if __name__ == "__main__":
    print(get_content("What are the different levels in taxonomy?"))
