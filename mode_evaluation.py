import os
import shutil

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

train_df = pd.read_csv('data/train/fake_news_small_train.csv')
test_df = pd.read_csv('data/test/fake_news_small_test.csv')

train_df['label'] = train_df['label'].astype(float)
test_df['label'] = test_df['label'].astype(float)

model = tf.keras.models.load_model("./model1/fake_bert")

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

optimizer = Adam(
    learning_rate=5e-05,
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

examples = [
    """The Biden administration is preparing a massive spending proposal on infrastructure and other domestic priorities like child care and drug costs that could put fights over hot-button issues like climate change and taxes front and center.A source familiar with the plans confirmed that administration officials are eyeing $3 trillion as the topline figure for its Build Back Better jobs and infrastructure proposal, though they cautioned talks are fluid and the final number could change. The sweeping package would constitute the White House’s follow-up to the $1.9 trillion economic relief measure signed into law earlier this month.

The new package is expected to be split into two separate bills. The first would focus on infrastructure, with spending on manufacturing and climate change measures, broadband and 5G, and the nation’s roads and bridges.

The other measure would include funds for pre-K programs, free community college tuition, child tax credits and health care subsidies, according to multiple reports.""",
    """London (CNN)London police released new details Tuesday about a shooting that left British Black Lives Matter activist Sasha Johnson fighting for her life in hospital, saying Johnson was shot by a group of four men at a party but that it did not appear to be a targeted attack.
The 27-year-old mother of three has been in critical condition in hospital since she was shot in the head at a house party in south London on Sunday.
"Around 3 a.m. local time on Sunday morning, a group of four black males dressed in dark colored clothing entered the garden of the property and discharged a firearm," London's Metropolitan police Commander Alison Heydari said in a statement Tuesday.

Heydari said that police were not aware of any threats made against Johnson prior to the incident.

"We are aware of Sasha's involvement in the Black Lives Matter movement in the UK and I understand the concern this will cause to some communities -- however I wish to stress that at this time there is nothing to suggest Sasha was the victim of a targeted attack," she said. """,
    """COLUMBUS, IN—Unfazed by the public swimming pool, local 7-year-old Logan Dixon told reporters Monday that he had seen way deeper deep ends. “Give me a break, what is this, five feet or something?” said Dixon, whose wisdom and courage left witnesses awestruck as he described how the public swimming pool’s depths paled in comparison to the those he had experienced at summer camp last year and at his friend Hunter’s house. “Sure, it’s deep by most people’s standards, but I’ve seen some things you wouldn’t believe. I’ve been in pools where I couldn’t even touch the bottom, not even on my tippy toes. Trust me, this is nothing.” At press time, Dixon was clinging to a foam pool noodle for dear life."""
]

results = tf.sigmoid(model(tf.constant(examples)))

for result in results:
    print(result[0])