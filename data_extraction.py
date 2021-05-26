import os
import shutil
import json

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

import csv

source_labels = dict()

with open('labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            source_labels[row[0]] = row[1]
            line_count += 1
    print(f'Processed {line_count} lines.')

count = 0
data = []
directory = 'nela-gt-2020/newsdata'
for file_name in os.listdir(directory):
    file = open(os.path.join(directory, file_name))
    data.extend(json.load(file))
    count += 1
    print(f'loaded {count} files')

count_fake = 0
count_true = 0
extracted_data = []
for datum in data:
    source = datum['source']
    if source in source_labels:
        if source_labels[source] == '0':
            count_true += 1
            extracted_data.append(
                {
                    'content': datum['content'],
                    'label': 0
                }
            )
        elif source_labels[source] == '2':
            count_fake += 1
            extracted_data.append(
                {
                    'content': datum['content'],
                    'label': 1
                }
            )

print(f'there are {count_fake} fake articles and {count_true} reliable ones')

keys = extracted_data[0].keys()
with open('fake_news.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(extracted_data)