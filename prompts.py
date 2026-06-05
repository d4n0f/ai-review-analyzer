"""All LLM prompts for the review analysis pipeline."""

import json


def get_preprocess_prompt(raw_text: str) -> str:
    return f"""Te egy adattisztító szakértő vagy. Feladatod webshop vásárlói vélemények előfeldolgozása.

Lépések (Chain-of-Thought):
1. Először azonosítsd az egyes review-kat (sorhatárok, számozás vagy üres sorok alapján).
2. Távolítsd el az irreleváns tartalmakat (pl. ismétlések, üres sorok).
3. Normalizáld a szövegeket (kis-nagybetű, extra szóközök).
4. Számold meg hány review-t azonosítottál.

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Példa bemenet:
"Nagyon jó termék!! Ajánlom mindenkinek.
rossz minőség, nem veszem meg többé
Gyors szállítás de drága"

Példa kimenet:
{{
  "reviews": [
    "Nagyon jó termék! Ajánlom mindenkinek.",
    "Rossz minőség, nem veszem meg többé.",
    "Gyors szállítás, de drága."
  ],
  "count": 3
}}

Feldolgozandó szöveg:
{raw_text}
"""


def get_topic_extraction_prompt(reviews: list) -> str:
    reviews_text = "\n".join([f"- {r}" for r in reviews])
    return f"""Te egy senior termékanalitikus vagy, aki webshop véleményekből témákat azonosít.

Lépések (Chain-of-Thought):
1. Először olvasd el az összes review-t.
2. Azonosítsd a visszatérő témaköröket (pl. szállítás, minőség, ár, ügyfélszolgálat, csomagolás).
3. Minden review-hoz rendeld hozzá a releváns témákat.
4. Számold meg melyik téma hányszor jelenik meg.

Few-shot példák:

Review: "Gyorsan megérkezett, jól volt csomagolva"
Témák: ["szállítás", "csomagolás"]

Review: "Drága de megéri, kiváló minőség"
Témák: ["ár", "minőség"]

Review: "Az ügyfélszolgálat segítőkész volt, de a visszaküldés bonyolult"
Témák: ["ügyfélszolgálat", "visszaküldés"]

Review: "A weboldal nehezen kezelhető, de a termék szép"
Témák: ["weboldal/UX", "termék megjelenés"]

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Kimenet formátuma:
{{
  "topics_per_review": [
    {{"review_index": 0, "topics": ["téma1", "téma2"]}},
    ...
  ],
  "all_topics": ["téma1", "téma2", ...],
  "topic_frequency": {{"téma1": 3, "téma2": 1, ...}}
}}

Review-k:
{reviews_text}
"""


def get_sentiment_prompt(reviews: list) -> str:
    reviews_text = "\n".join([f"{i}. {r}" for i, r in enumerate(reviews)])
    return f"""Te egy sentiment elemző szakértő vagy. Magyar vásárlói véleményeket értékelsz.

Lépések (Chain-of-Thought):
1. Először olvasd el a review-t teljes egészében.
2. Azonosítsd a pozitív jelzőket, kifejezéseket.
3. Azonosítsd a negatív jelzőket, kifejezéseket.
4. Döntsd el az összesített sentimentet: "pozitív", "negatív" vagy "semleges".
5. Adj egy 1-5 skálájú pontszámot (1=nagyon negatív, 5=nagyon pozitív).
6. Adj rövid magyarázatot a döntéshez.

Few-shot példák:

Review: "Fantasztikus termék, mindenkinek ajánlom! Gyors szállítás."
Elemzés: sentiment="pozitív", score=5, reason="Erős pozitív jelzők: fantasztikus, ajánlom. Gyors szállítás is pozitív."

Review: "Nem vagyok elégedett, rossz minőség és drága is."
Elemzés: sentiment="negatív", score=2, reason="Negatív jelzők: nem elégedett, rossz minőség. Ár is negatív."

Review: "Megkaptam a csomagot. Semmi különös."
Elemzés: sentiment="semleges", score=3, reason="Nincs erős pozitív vagy negatív tartalom."

Review: "Jó a termék, de a szállítás lassú volt."
Elemzés: sentiment="semleges", score=3, reason="Vegyes: pozitív termék, negatív szállítás."

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Kimenet formátuma:
{{
  "sentiments": [
    {{
      "review_index": 0,
      "sentiment": "pozitív",
      "score": 4,
      "reason": "rövid magyarázat",
      "positive_keywords": ["szó1"],
      "negative_keywords": []
    }},
    ...
  ],
  "overall_sentiment": "pozitív",
  "average_score": 3.8
}}

Review-k:
{reviews_text}
"""


def get_grouping_prompt(topics_data: dict, sentiments_data: dict, reviews: list) -> str:
    return f"""Te egy üzleti elemző vagy, aki vásárlói visszajelzéseket csoportosít.

Lépések (Chain-of-Thought):
1. Tekintsd át az azonosított témákat és azok gyakoriságát.
2. Csoportosítsd a hasonló témákat tematikus kategóriákba.
3. Minden csoporthoz rendeld hozzá az érintett review-k indexeit.
4. Határozd meg a csoport általános sentimentjét.
5. Emeld ki a legjellemzőbb pozitív és negatív mintákat.

Few-shot példa:
Ha témák: ["szállítás", "kézbesítés", "csomagolás"] → ezek mind a "Logisztika" csoportba kerülnek.
Ha témák: ["ár", "drágaság", "ár-érték arány"] → ezek az "Árazás" csoportba kerülnek.

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Kimenet formátuma:
{{
  "groups": [
    {{
      "group_name": "Logisztika",
      "topics": ["szállítás", "csomagolás"],
      "review_indices": [0, 2, 4],
      "dominant_sentiment": "pozitív",
      "summary": "A szállítás általában gyors, a csomagolás megfelelő."
    }},
    ...
  ],
  "total_groups": 3
}}

Témák adatok:
{json.dumps(topics_data, ensure_ascii=False, indent=2)}

Sentiment adatok összesítve:
overall_sentiment: {sentiments_data.get('overall_sentiment', 'semleges')}
average_score: {sentiments_data.get('average_score', 3)}

Feldolgozott review-k száma: {len(reviews)}
"""


def get_summary_prompt(groups: list, sentiments_data: dict, reviews: list) -> str:
    reviews_sample = reviews[:5] if len(reviews) > 5 else reviews
    reviews_text = "\n".join([f"- {r}" for r in reviews_sample])
    return f"""Te egy C-level üzleti tanácsadó vagy, aki executive summary-t ír vásárlói visszajelzésekből.

Lépések (Chain-of-Thought):
1. Tekintsd át a csoportosított témaköröket és azok sentimentjét.
2. Azonosítsd a 2-3 legfontosabb pozitív erősséget.
3. Azonosítsd a 2-3 legkritikusabb problémát.
4. Fogalmazd meg a fő üzenetet 2-3 mondatban.
5. Add meg az összesített vásárlói elégedettség szintjét százalékban.

Stílus: tömör, üzleti, adatvezérelt. Kerüld a terjengős fogalmazást.

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Kimenet formátuma:
{{
  "executive_summary": "2-3 mondatos összefoglaló",
  "key_strengths": ["erősség1", "erősség2"],
  "key_issues": ["probléma1", "probléma2"],
  "satisfaction_rate": 72,
  "review_count": {len(reviews)}
}}

Témacsoport adatok:
{json.dumps(groups, ensure_ascii=False, indent=2)}

Általános sentiment: {sentiments_data.get('overall_sentiment', 'semleges')}
Átlagos pontszám (1-5): {sentiments_data.get('average_score', 3)}

Minta review-k:
{reviews_text}
"""


def get_action_items_prompt(summary_data: dict, groups: list) -> str:
    return f"""Te egy tapasztalt e-commerce menedzser vagy, aki konkrét fejlesztési javaslatokat fogalmaz meg.

Lépések (Chain-of-Thought):
1. Tekintsd át a kulcsproblémákat és az erősségeket.
2. Minden problémához fogalmazz meg 1-2 konkrét, megvalósítható akciót.
3. Prioritizálj: magas/közepes/alacsony prioritás.
4. Becsüld meg a várható hatást: magas/közepes/alacsony.
5. Adj meg hozzávetőleges határidőt: rövid táv (1-4 hét), közép táv (1-3 hónap), hosszú táv (3+ hónap).

Few-shot példák:

Probléma: "Lassú szállítás"
Akció: "Váltás gyorsabb futárszolgálatra vagy express opció bevezetése"
Prioritás: magas, Hatás: magas, Határidő: rövid táv

Probléma: "Rossz termékkép minőség"
Akció: "Professzionális termékfotózás elvégzése a TOP 50 termékre"
Prioritás: közepes, Hatás: közepes, Határidő: közép táv

FONTOS: Csak valid JSON-t adj vissza, semmi mást!

Kimenet formátuma:
{{
  "action_items": [
    {{
      "title": "Akció neve",
      "description": "Részletes leírás",
      "priority": "magas",
      "expected_impact": "magas",
      "timeline": "rövid táv",
      "related_issue": "kapcsolódó probléma"
    }},
    ...
  ],
  "quick_wins": ["azonnal megvalósítható dolog1", "azonnal megvalósítható dolog2"]
}}

Összefoglaló adatok:
{json.dumps(summary_data, ensure_ascii=False, indent=2)}

Témacsoport adatok:
{json.dumps(groups, ensure_ascii=False, indent=2)}
"""
