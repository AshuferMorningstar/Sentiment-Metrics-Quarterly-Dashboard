import kagglehub
import pandas as pd

SAMPLE_SIZE = 5000

# Download Sentiment140 dataset using kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")

# 1. Load data - balanced 5K-row demo sample
source = pd.read_csv(
    f"{path}/training.1600000.processed.noemoticon.csv",
    usecols=['target', 'id', 'date', 'flag', 'user', 'text'],
    encoding='latin-1',
    header=None,
    names=['target', 'id', 'date', 'flag', 'user', 'text'],
)
negative_sample = source[source['target'] == 0].sample(n=SAMPLE_SIZE // 2, random_state=42)
positive_sample = source[source['target'] == 4].sample(n=SAMPLE_SIZE // 2, random_state=42)
df = pd.concat([negative_sample, positive_sample], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Data validation - track completeness
initial_rows = len(df)  
df = df.drop_duplicates()
df = df.dropna(subset=['text'])
final_rows = len(df)
data_completeness = round((final_rows/initial_rows)*100, 2)

# 3. Add new features for analysis
df['sentiment_label'] = df['target'].map({0:'Negative', 4:'Positive'})
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# 4. Calculate metrics for dashboard
metrics = df.groupby('sentiment_label').agg(
    volume=('text','count'),
    avg_length=('text_length','mean'),
    avg_words=('word_count','mean')
).reset_index()
metrics['completeness'] = data_completeness

# 5. Save outputs for next steps
df.to_csv('clean_sentiment.csv', index=False)
metrics.to_csv('metrics.csv', index=False)

# Print validation + metrics
print(f"Data completeness: {data_completeness}%")
print(metrics)