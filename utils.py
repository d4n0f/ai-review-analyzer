"""Utility helpers: chart generation and JSON export."""

import json
import plotly.graph_objects as go
import plotly.express as px


def build_topic_frequency_chart(topic_frequency: dict) -> go.Figure:
    if not topic_frequency:
        return None
    topics = list(topic_frequency.keys())
    counts = list(topic_frequency.values())
    # Sort descending
    pairs = sorted(zip(counts, topics), reverse=True)
    counts, topics = zip(*pairs)

    fig = px.bar(
        x=list(counts),
        y=list(topics),
        orientation="h",
        labels={"x": "Előfordulás", "y": "Téma"},
        title="Témák gyakorisága",
        color=list(counts),
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=max(300, len(topics) * 40),
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def build_sentiment_pie_chart(sentiments_data: dict) -> go.Figure:
    sentiments = sentiments_data.get("sentiments", [])
    if not sentiments:
        return None

    counts = {"pozitív": 0, "negatív": 0, "semleges": 0}
    for s in sentiments:
        label = s.get("sentiment", "semleges").lower()
        if label in counts:
            counts[label] += 1
        else:
            counts["semleges"] += 1

    color_map = {"pozitív": "#4CAF50", "negatív": "#F44336", "semleges": "#9E9E9E"}
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    colors = [color_map[k] for k in labels]

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=values, marker_colors=colors, hole=0.4)]
    )
    fig.update_layout(
        title="Sentiment megoszlás",
        height=300,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    return fig


def sentiment_badge(sentiment: str) -> str:
    mapping = {
        "pozitív": "🟢",
        "negatív": "🔴",
        "semleges": "🟡",
    }
    return mapping.get(sentiment.lower(), "⚪")


def priority_badge(priority: str) -> str:
    mapping = {
        "magas": "🔴 Magas",
        "közepes": "🟡 Közepes",
        "alacsony": "🟢 Alacsony",
    }
    return mapping.get(priority.lower(), priority)


def export_results_json(results: dict) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)
