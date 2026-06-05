# AI Review Analyzer

Multi-step AI pipeline for analyzing Hungarian e-commerce reviews — built with Streamlit and Groq (LLaMA 3.3 70B).

## What it does

6 sequential LLM calls process raw review text end-to-end:

| Step | Description |
|------|-------------|
| 1. Preprocessing | Split and normalize raw review text |
| 2. Topic extraction | Identify recurring topics (shipping, price, quality…) |
| 3. Sentiment analysis | Positive / negative / neutral + 1–5 score per review |
| 4. Grouping | Cluster similar topics into categories |
| 5. Executive summary | Key strengths, issues, satisfaction rate |
| 6. Action items | Prioritized improvement tasks with timelines |

**Stack:** Python · Streamlit · Groq API · Plotly · python-dotenv

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_...
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
streamlit run app.py
```

App runs at `http://localhost:8501` by default.

## Project structure

```
├── app.py          # Streamlit UI and progress tracking
├── processing.py   # 6-step pipeline orchestrator
├── prompts.py      # LLM prompts (Chain-of-Thought + few-shot)
├── llm.py          # Groq API wrapper, lazy singleton client
├── utils.py        # Plotly charts, badges, JSON export
└── requirements.txt
```

## LLM parameters

Configurable at runtime via the sidebar:

- **Temperature** (0–1): lower = more deterministic output
- **Max tokens** (256–2048): token limit per LLM call
- **Top-p** (0.1–1.0): nucleus sampling threshold

## License

This project is licensed under the [MIT License](LICENSE).
