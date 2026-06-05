"""Groq API wrapper with error handling."""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY nincs beállítva a .env fájlban")
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    top_p: float = 1.0,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Te egy precíz AI asszisztens vagy. "
                    "MINDIG csak valid JSON-t adsz vissza, soha nem írsz extra magyarázatot, "
                    "markdown kódblokk jelölőket (```), vagy bármilyen más szöveget a JSON-on kívül."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content
