import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv('fake_news_small.csv')

arr = [len(content.split()) for content in df['content']]

plt.hist(arr, bins=300)
plt.show()