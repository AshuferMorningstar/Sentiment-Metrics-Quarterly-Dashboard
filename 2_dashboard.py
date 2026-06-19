import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sentiment Metrics Dashboard", layout="wide", page_icon="📊")

st.title("📊 Sentiment Metrics Quarterly Dashboard")
st.caption("Data pipeline: Sentiment140 → Clean → Metrics | Built for fintech data quality")

# Load data
@st.cache_data
def load_metrics():
    data = pd.read_csv("metrics.csv")
    required_columns = {"sentiment_label", "volume", "avg_length", "avg_words"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"metrics.csv is missing columns: {', '.join(sorted(missing_columns))}")
    return data

df = load_metrics()

total_volume = int(df["volume"].sum())
weighted_avg_length = (df["avg_length"] * df["volume"]).sum() / total_volume if total_volume else 0
weighted_avg_words = (df["avg_words"] * df["volume"]).sum() / total_volume if total_volume else 0

chart_data = df.set_index("sentiment_label")[["volume"]]

# KPIs - top row
col1, col2, col3, col4 = st.columns(4)

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

st.divider()

# Charts - 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    st.bar_chart(chart_data)

with col2:
    st.subheader("Average Text Length and Word Count")
    st.bar_chart(df.set_index("sentiment_label")[ ["avg_length", "avg_words"] ])

st.divider()

# Data table
st.subheader("Metrics Table")
st.dataframe(df, use_container_width=True)

st.caption("Note: Data downloaded via kagglehub on runtime. Repo optimized to stay <100MB for GitHub limits.")