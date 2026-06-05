"""AI Review Elemző — Streamlit frontend."""

import json
import streamlit as st

from processing import run_pipeline
from utils import (
    build_topic_frequency_chart,
    build_sentiment_pie_chart,
    sentiment_badge,
    priority_badge,
    export_results_json,
)

SAMPLE_REVIEWS = """Nagyon gyorsan megérkezett a csomag, de a termék minősége hagy kívánni valót maga után. Nem vagyok elégedett.
Az ügyfélszolgálat segítőkész volt, gyorsan megoldották a problémámat. Ajánlom mindenkinek!
Drága a termék az értékéhez képest, de a csomagolás nagyon szép volt.
A weboldal nehezen kezelhető, sokáig tartott megtalálni a terméket. Maga a termék viszont kiváló minőségű.
Visszaküldési folyamat bonyolult és lassú. Két hetet vártam a visszatérítésre.
Fantasztikus termék, pontosan olyan mint a leírásban. Gyors szállítás, szép csomagolás. Nagyon elégedett vagyok!
Az ár megfelelő, de a szállítás késett 3 napot. Semmi magyarázat, semmi értesítés.
Kitűnő minőség, de drágán van árazva a versenytársakhoz képest."""

st.set_page_config(
    page_title="AI Review Elemző",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Review Elemző")
st.caption("Magyar vélemények többlépéses AI elemzése")

# ── Sidebar: parameters ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Paraméterek")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Alacsonyabb = konzisztensebb, determinisztikusabb output",
    )

    max_tokens = st.slider(
        "Max tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=128,
        help="Egy LLM hívás maximális token korlátja",
    )

    top_p = st.slider(
        "Top-p (nucleus sampling)",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Nucleus sampling valószínűségi küszöb",
    )

    st.divider()
    st.markdown("**Pipeline lépések:**")
    st.markdown("""
1. 🔄 Input előfeldolgozás
2. 🏷️ Téma azonosítás
3. 💬 Sentiment elemzés
4. 🗂️ Témák csoportosítása
5. 📝 Executive summary
6. ✅ Akció javaslatok
""")

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("📝 Review-k megadása")

col_input, col_sample = st.columns([3, 1])
with col_sample:
    if st.button("Minta betöltése", use_container_width=True):
        st.session_state["review_text"] = SAMPLE_REVIEWS
        st.rerun()

review_text = st.text_area(
    "Írd be a véleményeket (soronként egy review, vagy bekezdésenként):",
    value=st.session_state.get("review_text", ""),
    height=200,
    placeholder="Pl.: Gyors szállítás, de rossz minőség.\nNagyon elégedett vagyok, ajánlom!",
)

run_btn = st.button("🚀 Elemzés indítása", type="primary", use_container_width=True)

# ── Validation & Run ─────────────────────────────────────────────────────────
if run_btn:
    if not review_text or not review_text.strip():
        st.error("❌ Adj meg legalább egy review-t az elemzéshez!")
        st.stop()

    progress_placeholder = st.empty()
    status_bar = st.progress(0)
    step_labels = [
        "Előfeldolgozás...",
        "Témák azonosítása...",
        "Sentiment elemzés...",
        "Témák csoportosítása...",
        "Összefoglaló generálása...",
        "Akció javaslatok generálása...",
    ]
    step_count = [0]

    def update_progress(msg: str):
        idx = next((i for i, l in enumerate(step_labels) if l == msg), step_count[0])
        step_count[0] = idx + 1
        progress_placeholder.info(f"**{step_count[0]}/6** — {msg}")
        status_bar.progress(step_count[0] / 6)

    try:
        results = run_pipeline(
            raw_text=review_text,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            progress_cb=update_progress,
        )
        st.session_state["results"] = results
        progress_placeholder.success("✅ Elemzés kész!")
        status_bar.progress(1.0)

    except ValueError as e:
        progress_placeholder.empty()
        status_bar.empty()
        st.error(f"❌ Pipeline hiba: {e}")
        st.stop()
    except Exception as e:
        progress_placeholder.empty()
        status_bar.empty()
        err_msg = str(e)
        if "api" in err_msg.lower() or "groq" in err_msg.lower() or "key" in err_msg.lower():
            st.error(f"❌ API hiba: {err_msg}")
        else:
            st.error(f"❌ Váratlan hiba: {err_msg}")
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results = st.session_state["results"]
    reviews = results["reviews"]
    topics_data = results["topics"]
    sentiments_data = results["sentiments"]
    groups_data = results["groups"]
    summary_data = results["summary"]
    actions_data = results["actions"]

    st.divider()

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.subheader("📊 Összefoglaló")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Feldolgozott review-k", len(reviews))
    kpi2.metric(
        "Általános sentiment",
        sentiments_data.get("overall_sentiment", "—").capitalize(),
    )
    avg_score = sentiments_data.get("average_score", 0)
    kpi3.metric("Átlagos pontszám", f"{avg_score:.1f} / 5")
    kpi4.metric(
        "Elégedettség",
        f"{summary_data.get('satisfaction_rate', '—')}%",
    )

    with st.expander("📋 Executive summary", expanded=True):
        st.write(summary_data.get("executive_summary", "—"))

        col_str, col_iss = st.columns(2)
        with col_str:
            st.markdown("**Fő erősségek:**")
            for s in summary_data.get("key_strengths", []):
                st.markdown(f"- ✅ {s}")
        with col_iss:
            st.markdown("**Fő problémák:**")
            for p in summary_data.get("key_issues", []):
                st.markdown(f"- ⚠️ {p}")

    # ── Topics ────────────────────────────────────────────────────────────────
    st.subheader("🏷️ Azonosított témák")

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig = build_topic_frequency_chart(topics_data.get("topic_frequency", {}))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nincs elegendő adat a diagramhoz.")

    with col_table:
        freq = topics_data.get("topic_frequency", {})
        if freq:
            sorted_topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            st.markdown("**Téma | Előfordulás**")
            for topic, count in sorted_topics:
                st.markdown(f"- **{topic}**: {count}×")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    st.subheader("💬 Sentiment elemzés")

    col_pie, col_list = st.columns([2, 3])
    with col_pie:
        pie = build_sentiment_pie_chart(sentiments_data)
        if pie:
            st.plotly_chart(pie, use_container_width=True)

    with col_list:
        with st.expander("Review-k sentiment bontásban", expanded=True):
            for i, s in enumerate(sentiments_data.get("sentiments", [])):
                badge = sentiment_badge(s.get("sentiment", "semleges"))
                review_text_short = reviews[i][:100] + "..." if len(reviews[i]) > 100 else reviews[i]
                score = s.get("score", "—")
                reason = s.get("reason", "")
                st.markdown(
                    f"{badge} **{i+1}.** _{review_text_short}_  \n"
                    f"Pontszám: **{score}/5** — {reason}"
                )
                st.divider()

    # ── Topic Groups ──────────────────────────────────────────────────────────
    st.subheader("🗂️ Témakategóriák")

    groups = groups_data.get("groups", [])
    if groups:
        cols = st.columns(min(len(groups), 3))
        for i, group in enumerate(groups):
            with cols[i % 3]:
                badge = sentiment_badge(group.get("dominant_sentiment", "semleges"))
                with st.expander(
                    f"{badge} {group.get('group_name', 'Csoport')} ({len(group.get('review_indices', []))} review)",
                    expanded=True,
                ):
                    st.write(group.get("summary", ""))
                    topics_in_group = group.get("topics", [])
                    if topics_in_group:
                        st.markdown("**Témák:** " + ", ".join(f"`{t}`" for t in topics_in_group))
    else:
        st.info("Nem sikerült témakategóriákat azonosítani.")

    # ── Action Items ──────────────────────────────────────────────────────────
    st.subheader("✅ Ajánlott akciók")

    action_items = actions_data.get("action_items", [])
    quick_wins = actions_data.get("quick_wins", [])

    if quick_wins:
        with st.expander("⚡ Azonnali quick wins", expanded=True):
            for qw in quick_wins:
                st.markdown(f"- 🚀 {qw}")

    if action_items:
        for item in action_items:
            priority_label = priority_badge(item.get("priority", "közepes"))
            with st.expander(
                f"{priority_label} — {item.get('title', 'Akció')}",
                expanded=False,
            ):
                st.write(item.get("description", ""))
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Várható hatás", item.get("expected_impact", "—").capitalize())
                col_b.metric("Határidő", item.get("timeline", "—"))
                col_c.metric("Kapcsolódó probléma", item.get("related_issue", "—"))
    else:
        st.info("Nem sikerült akció javaslatokat generálni.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Exportálás")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        json_str = export_results_json(results)
        st.download_button(
            label="📥 Teljes elemzés letöltése (JSON)",
            data=json_str,
            file_name="review_analysis.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_exp2:
        with st.expander("🔍 Nyers JSON megtekintése"):
            st.json(results)
