"""6-step review analysis pipeline. Each step is a separate LLM call."""

import json
import re
from typing import Callable, Optional

from llm import call_llm
from prompts import (
    get_preprocess_prompt,
    get_topic_extraction_prompt,
    get_sentiment_prompt,
    get_grouping_prompt,
    get_summary_prompt,
    get_action_items_prompt,
)


def _parse_json(raw: str) -> dict:
    """Extract and parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


def preprocess_reviews(
    raw_text: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 1: Clean and split raw review text into individual reviews."""
    if progress_cb:
        progress_cb("Előfeldolgozás...")

    prompt = get_preprocess_prompt(raw_text)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "reviews" not in result:
            raise ValueError("Hiányzó 'reviews' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Előfeldolgozás JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def extract_topics(
    reviews: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 2: Extract topics from each review."""
    if progress_cb:
        progress_cb("Témák azonosítása...")

    prompt = get_topic_extraction_prompt(reviews)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "topic_frequency" not in result:
            raise ValueError("Hiányzó 'topic_frequency' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Témaextrakció JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def analyze_sentiment(
    reviews: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 3: Sentiment analysis for each review."""
    if progress_cb:
        progress_cb("Sentiment elemzés...")

    prompt = get_sentiment_prompt(reviews)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "sentiments" not in result:
            raise ValueError("Hiányzó 'sentiments' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Sentiment elemzés JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def group_topics(
    topics_data: dict,
    sentiments_data: dict,
    reviews: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 4: Group similar topics into categories."""
    if progress_cb:
        progress_cb("Témák csoportosítása...")

    prompt = get_grouping_prompt(topics_data, sentiments_data, reviews)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "groups" not in result:
            raise ValueError("Hiányzó 'groups' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Csoportosítás JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def generate_summary(
    groups: list,
    sentiments_data: dict,
    reviews: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 5: Generate executive summary."""
    if progress_cb:
        progress_cb("Összefoglaló generálása...")

    prompt = get_summary_prompt(groups, sentiments_data, reviews)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "executive_summary" not in result:
            raise ValueError("Hiányzó 'executive_summary' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Összefoglaló JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def generate_actions(
    summary_data: dict,
    groups: list,
    temperature: float,
    max_tokens: int,
    top_p: float,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Step 6: Generate concrete action items."""
    if progress_cb:
        progress_cb("Akció javaslatok generálása...")

    prompt = get_action_items_prompt(summary_data, groups)
    raw = call_llm(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)

    try:
        result = _parse_json(raw)
        if "action_items" not in result:
            raise ValueError("Hiányzó 'action_items' mező")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Akció generálás JSON hiba: {e}\nNyers válasz: {raw[:300]}")


def run_pipeline(
    raw_text: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Run all 6 pipeline steps and return aggregated results."""
    # Step 1
    preprocessed = preprocess_reviews(raw_text, temperature, max_tokens, top_p, progress_cb)
    reviews = preprocessed["reviews"]

    if not reviews:
        raise ValueError("Nem sikerült review-kat azonosítani a szövegből.")

    # Step 2
    topics_data = extract_topics(reviews, temperature, max_tokens, top_p, progress_cb)

    # Step 3
    sentiments_data = analyze_sentiment(reviews, temperature, max_tokens, top_p, progress_cb)

    # Step 4
    groups_data = group_topics(topics_data, sentiments_data, reviews, temperature, max_tokens, top_p, progress_cb)

    # Step 5
    summary_data = generate_summary(
        groups_data["groups"], sentiments_data, reviews, temperature, max_tokens, top_p, progress_cb
    )

    # Step 6
    actions_data = generate_actions(
        summary_data, groups_data["groups"], temperature, max_tokens, top_p, progress_cb
    )

    return {
        "reviews": reviews,
        "topics": topics_data,
        "sentiments": sentiments_data,
        "groups": groups_data,
        "summary": summary_data,
        "actions": actions_data,
    }
