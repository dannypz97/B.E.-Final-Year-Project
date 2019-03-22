from classifier import SentenceClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

import re

classifier = SentenceClassifier()
mydb, cursor = classifier.connect_to_db()

cursor.execute("select tags from compiled")
tags = list(re.split(',\s*', tag[0]) for tag in cursor.fetchall())

for i, ele in enumerate(tags):
    for j, tag in enumerate(ele):
        if len(tag)==0 or tag==',':
            del tags[i][j]

mlb = MultiLabelBinarizer()
mlb.fit(tags)

tags_encoded = mlb.fit_transform(tags)

for tag in tags_encoded:
    print(tag)

print(mlb.classes_)
