import pandas as pd
import numpy as np

df = pd.read_csv('fake_news.csv')

print(type(df['label'][0]))
print(df['label'][0])

fake_df = df[df['label'] == 1]
true_df = df[df['label'] == 0]

print(len(fake_df))
print(len(true_df))

fake_df = fake_df.sample(n=5000)
true_df = true_df.sample(n=5000)

print(len(fake_df))
print(len(true_df))

extracted_df = pd.concat([fake_df, true_df])
extracted_df = extracted_df.sample(frac=1)

print(len(extracted_df))

extracted_df.to_csv('fake_news_small.csv', index=False)