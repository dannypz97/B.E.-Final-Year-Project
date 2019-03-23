from classifier import SentenceClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
import pandas as pd
import re
import json

classifier = SentenceClassifier()
mydb, cursor = classifier.connect_to_db()

cursor.execute("select question from " + "compiled")  # load questions from db
questions = list(str(x[0]) for x in cursor.fetchall())

cursor.execute("select tags from compiled")
tags = list(re.split(',\s*', tag[0]) for tag in cursor.fetchall())


print(questions)
print(tags)

with open('./data/stack-exchange/Law.json','r', encoding='utf8') as f:
    datastore = json.load(f)
    questions = []
    tags = []
    for row in datastore:
        questions.append(row['question'])
        tags.append(row['tags'].split(', '))

    print(questions)
    print(tags)


