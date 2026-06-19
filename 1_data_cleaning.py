import kagglehub
import pandas as pd
import re

SAMPLE_SIZE = 5000
CLASS_ORDER = ['Negative', 'Neutral', 'Positive']

POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'happy', 'win', 'wins', 'winning',
    'best', 'fantastic', 'positive', 'enjoy', 'joy', 'beautiful', 'brilliant', 'nice', 'perfect',
    'cool', 'wonderful', 'pleased', 'delight', 'success', 'successful', 'like'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'worst', 'poor', 'negative', 'fail',
    'fails', 'failing', 'failure', 'ugly', 'horrible', 'pain', 'annoyed', 'disappointed', 'nasty',
    'upset', 'unhappy', 'depressing', 'tired', 'sick', 'broken'
}


def label_sentiment(text: str) -> tuple[str, int]:
    tokens = re.findall(r"[a-z']+", text.lower())
    positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
    negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)
    score = positive_hits - negative_hits
    if score > 0:
        return 'Positive', score
    if score < 0:
        return 'Negative', score
    return 'Neutral', score

# Download Sentiment140 dataset using kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")

# 1. Load data - 5K-row demo sample with three-way sentiment labels
source = pd.read_csv(
    f"{path}/training.1600000.processed.noemoticon.csv",
    usecols=['target', 'id', 'date', 'flag', 'user', 'text'],
    encoding='latin-1',
    header=None,
    names=['target', 'id', 'date', 'flag', 'user', 'text'],
)
df = source.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# 2. Data validation - track completeness
initial_rows = len(df)  
df = df.drop_duplicates()
df = df.dropna(subset=['text'])
final_rows = len(df)
data_completeness = round((final_rows/initial_rows)*100, 2)

# 3. Add new features for analysis
sentiment_results = df['text'].apply(label_sentiment)
df['sentiment_label'] = sentiment_results.apply(lambda item: item[0])
df['sentiment_score'] = sentiment_results.apply(lambda item: item[1])
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# 4. Calculate metrics for dashboard
metrics = df.groupby('sentiment_label').agg(
    volume=('text','count'),
    avg_length=('text_length','mean'),
    avg_words=('word_count','mean')
).reset_index()
metrics['completeness'] = data_completeness
metrics['sentiment_label'] = pd.Categorical(metrics['sentiment_label'], categories=CLASS_ORDER, ordered=True)
metrics = metrics.set_index('sentiment_label').reindex(CLASS_ORDER).reset_index()
metrics['completeness'] = metrics['completeness'].fillna(data_completeness)
metrics[['volume', 'avg_length', 'avg_words']] = metrics[['volume', 'avg_length', 'avg_words']].fillna(0)
metrics['sentiment_label'] = metrics['sentiment_label'].astype(str)

# 5. Save outputs for next steps
df.to_csv('clean_sentiment.csv', index=False)
metrics.to_csv('metrics.csv', index=False)

# Print validation + metrics
print(f"Data completeness: {data_completeness}%")
print(metrics)