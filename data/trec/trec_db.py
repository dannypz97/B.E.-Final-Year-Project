'''
This program takes questions from the TREC-6  train set (train_5500.label.txt) and test set (TREC_10.label.txt),
preprocesses them and then inserts the questions and associated labels into a MySQL database.
'''
from json import encoder

import mysql.connector
import re

label_set = set()
def clean_str(string):
    """
    Cleans each string and convert to lower case.
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().rstrip()

def data_to_db(TRAIN_DATA_PATH='./train_5500.label.txt', TEST_DATA_PATH='./TREC_10.label.txt'):
    """
    Load questions and fine category labels to database.
    """
    dataset1 = list(open(TRAIN_DATA_PATH, encoding='utf-8', errors='replace').readlines())
    dataset2 = list(open(TEST_DATA_PATH, encoding='utf-8', errors='replace').readlines())
    dataset = dataset1 + dataset2

    x = [s.split(" ", 1)[1] for s in dataset]
    y = [s.split(" ")[0] for s in dataset]



    broad_classes = {'DESC':'description', 'ENTY':'entity', 'ABBR':'abbreviation','HUM':'human','NUM':'numeric', 'LOC':'location'}

    fine_classes = {'cremat':'creative', 'ind':'individual', 'def':'definition', 'exp':'expression',
                    'dist':'distance', 'lang':'language', 'gr':'group', 'termeq':'term', 'dismed':'medicine', 'mount':'mountain',
                    'tecmeth':'technique', 'volsize':'size', 'instru': 'instrument', 'abb': 'abbreviation', 'perc':'percentage', 'temp': 'temperature',
                    'ord':'order', 'veh':'vehicle', 'desc':'description'}

    y_processed = [] #labels after processing


    for labels in y:

        broad_class, fine_class = labels.split(':')

        broad_class = broad_classes[broad_class]
        '''
        if fine_class in fine_classes:
            fine_class = fine_classes[fine_class]
        '''
        label = 'spam'
        label_set.add(label)
        y_processed.append(label)
    return x, y_processed


mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()

sql1 = "INSERT INTO `Photography` (question, tags) VALUES (%s, %s)"
sql2 = 'insert into tags (`table`, `broad_label`, `fine_label`) values (%s, %s, %s)'

questions, tags = data_to_db()

iter = 0
for val in zip(questions, tags):
    if iter < 2000:
        if not ('photo' in val[0].lower() or 'camera' in val[0].lower() or 'light' in val[0].lower() or 'software' in val[0].lower() or 'filter' in val[0].lower()):
            print(val[0])
            iter = iter + 1
            cursor.execute(sql1, val)
mydb.commit()


'''
for i in range(0, 50):
    for j in range(i+1, len(questions)):
        question = questions[i] + " and " + questions[j]
        tag = tags[i] + "," + tags[j]
        cursor.execute(sql1, (question, tag))

        question = questions[j] + " and " + questions[i]
        tag = tags[j] + "," + tags[i]
        cursor.execute(sql1, (question, tag))
    print("DONE!!")
    mydb.commit()


for tag in label_set:

    cursor.execute(sql2, ('trec', tag, '-'))
mydb.commit()
'''