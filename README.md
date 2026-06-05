# AI Review Elemző

Magyar vélemények többlépéses AI elemzése — Streamlit + Groq (LLaMA 3.3 70B).

## Mit csinál?

6 egymást követő LLM-hívásból álló pipeline dolgozza fel a nyers véleményszövegeket:

| Lépés | Leírás |
|-------|--------|
| 1. Előfeldolgozás | Review-k szétválasztása, normalizálása |
| 2. Témaextrakció | Visszatérő témák azonosítása (szállítás, ár, minőség…) |
| 3. Sentiment elemzés | Pozitív / negatív / semleges, 1–5 pontszám review-nként |
| 4. Csoportosítás | Hasonló témák kategóriákba rendezése |
| 5. Executive summary | Fő erősségek, problémák, elégedettségi ráta |
| 6. Akció javaslatok | Prioritizált fejlesztési teendők határidőkkel |

**Technológiák:** Python · Streamlit · Groq API · Plotly · python-dotenv

## Beállítás

### 1. Függőségek telepítése

```bash
pip install -r requirements.txt
```

### 2. API kulcs

```bash
cp .env.example .env
```

Szerkeszd a `.env` fájlt, és add meg a Groq API kulcsodat:

```
GROQ_API_KEY=gsk_...
```

### 3. Indítás

```bash
streamlit run app.py
```

## Projekt struktúra

```
├── app.py          # Streamlit UI és progress tracking
├── processing.py   # 6-lépéses pipeline orchestrátor
├── prompts.py      # Összes LLM prompt (Chain-of-Thought + few-shot)
├── llm.py          # Groq API wrapper, lazy singleton kliens
├── utils.py        # Plotly diagramok, badge-ek, JSON export
└── requirements.txt
```

## LLM paraméterek

A sidebar-ból futásidőben állítható:

- **Temperature** (0–1): alacsonyabb = determinisztikusabb output
- **Max tokens** (256–2048): egy hívás token limitje
- **Top-p** (0.1–1.0): nucleus sampling küszöb

## Kimenet

Az elemzés eredménye letölthető JSON formátumban, és megtekinthető közvetlenül az app-ban.
