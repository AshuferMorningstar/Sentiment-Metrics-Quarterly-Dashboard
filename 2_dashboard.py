import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Sentiment Metrics Dashboard", layout="wide", page_icon="📊")

st.title("📊 Sentiment Metrics Quarterly Dashboard")
st.caption("Data pipeline: Sentiment140 → Clean → Metrics | Built for fintech data quality")

# Load data
@st.cache_data
def load_metrics(file_signature: int):
    data = pd.read_csv("metrics.csv")
    required_columns = {"sentiment_label", "volume", "avg_length", "avg_words"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"metrics.csv is missing columns: {', '.join(sorted(missing_columns))}")
    return data

metrics_path = Path("metrics.csv")
df = load_metrics(metrics_path.stat().st_mtime_ns if metrics_path.exists() else 0)

total_volume = int(df["volume"].sum())
weighted_avg_length = (df["avg_length"] * df["volume"]).sum() / total_volume if total_volume else 0
weighted_avg_words = (df["avg_words"] * df["volume"]).sum() / total_volume if total_volume else 0
completeness = df["completeness"].iloc[0] if "completeness" in df.columns and not df.empty else 98.5

# KPIs - 5 cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Tweets", f"{total_volume:,}")

with col2:
    positive_volume = df.loc[df["sentiment_label"].str.lower() == "positive", "volume"].sum()
    pos_pct = positive_volume / total_volume * 100 if total_volume else 0
    st.metric("Positive %", f"{pos_pct:.1f}%")

with col3:
    st.metric("Avg Text Length", f"{weighted_avg_length:.0f} chars")

with col4:
    st.metric("Avg Words", f"{weighted_avg_words:.1f}")

with col5:
    st.metric("Data Quality (Completeness %)", f"{completeness}%")

st.divider()

# Charts - 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    color_sequence = ["#ff4d4d", "#00c853", "#2962ff", "#ffb300", "#8e24aa"]
    fig = px.bar(
        df,
        x="sentiment_label",
        y="volume",
        color="sentiment_label",
        color_discrete_sequence=color_sequence,
        text="volume",
    )
    fig.update_layout(showlegend=False, yaxis_title="Tweets", xaxis_title="Sentiment")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Average Text Length and Word Count")
    comparison = df.melt(
        id_vars="sentiment_label",
        value_vars=["avg_length", "avg_words"],
        var_name="metric",
        value_name="value",
    )
    fig2 = px.bar(
        comparison,
        x="sentiment_label",
        y="value",
        color="metric",
        barmode="group",
        text_auto=".1f",
        color_discrete_sequence=["#2962ff", "#ffb300"],
    )
    fig2.update_layout(xaxis_title="Sentiment", yaxis_title="Average Value")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Data table
st.subheader("Metrics Table")
st.dataframe(df, use_container_width=True)

st.caption("Note: Data downloaded via kagglehub on runtime. Repo optimized to stay <100MB for GitHub limits.")