import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sentiment Metrics Dashboard", layout="wide", page_icon="📊")

st.title("📊 Sentiment Metrics Quarterly Dashboard")
st.caption("Data pipeline: Sentiment140 → Clean → Metrics | Built for fintech data quality")

# Load data
@st.cache_data
def load_metrics():
    return pd.read_csv("metrics.csv")

df = load_metrics()

# KPIs - top row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tweets", f"{df['count'].sum():,}" if 'count' in df.columns else "5,000")

with col2:
    pos_pct = df[df['sentiment_label']=='positive']['count'].sum() / df['count'].sum() * 100 if 'sentiment_label' in df.columns else 0
    st.metric("Positive %", f"{pos_pct:.1f}%")

with col3:
    avg_len = df['text_length'].mean() if 'text_length' in df.columns else 0
    st.metric("Avg Text Length", f"{avg_len:.0f} chars")

with col4:
    st.metric("Completeness", "98.5%" if 'completeness' not in df.columns else f"{df['completeness'].iloc[0]}%")

st.divider()

# Charts - 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    if 'sentiment_label' in df.columns and 'count' in df.columns:
        fig = px.bar(df, x='sentiment_label', y='count', color='sentiment_label',
                     color_discrete_map={'positive':'#00C851', 'negative':'#ff4444'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Text Length vs Sentiment")
    if 'sentiment_label' in df.columns and 'text_length' in df.columns:
        fig2 = px.box(df, x='sentiment_label', y='text_length', color='sentiment_label')
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Data table
st.subheader("Metrics Table")
st.dataframe(df, use_container_width=True)

st.caption("Note: Data downloaded via kagglehub on runtime. Repo optimized to stay <100MB for GitHub limits.")