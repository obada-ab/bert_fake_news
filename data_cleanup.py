import pandas as pd
import numpy as np

df = pd.read_csv('fake_news_small.csv')

print(df.columns)
indicies = []
i = 0

for content in df['content']:
    if type(content) is not str:
        indicies.append(i)
    else:
        print(i, max(0, len(content) - 2500))
        df.at[i, 'content'] = content[:2500]
    i += 1

df = df.drop(indicies)
df.to_csv('fake_news_small.csv', index=False)