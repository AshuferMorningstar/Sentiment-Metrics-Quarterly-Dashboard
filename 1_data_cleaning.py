import pandas as pd

# 1. Load data
df = pd.read_csv('sentiment140.csv', nrows=10000, encoding='latin-1', 
                 names=['target','id','date','flag','user','text'])

# 2. Data cleaning - validation
df = df.drop_duplicates()
df = df.dropna(subset=['text'])
initial_rows = 10000
final_rows = len(df)
data_completeness = round((final_rows/initial_rows)*100, 2)

# 3. Add new features
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

print(f"Data completeness: {data_completeness}%")
print(metrics)