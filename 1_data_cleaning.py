import kagglehub
import pandas as pd

# Download Sentiment140 dataset using kagglehub
path = kagglehub.dataset_download("kazanova/sentiment140")

# 1. Load data - 5K rows for demo
df = pd.read_csv(f"{path}/training.1600000.processed.noemoticon.csv", nrows=5000, encoding='latin-1', names=['target','id','date','flag','user','text'])

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

# 5. Save outputs for next steps
df.to_csv('clean_sentiment.csv', index=False)
metrics.to_csv('metrics.csv', index=False)

# Print validation + metrics
print(f"Data completeness: {data_completeness}%")
print(metrics)