import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('fake_news_small.csv')

train, test = train_test_split(df, stratify=df['label'], test_size=0.2)

train.to_csv('fake_news_small_train.csv', index=False)
test.to_csv('fake_news_small_test.csv', index=False)