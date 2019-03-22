'''
This program takes questions from the TREC-6  train set (train_5500.label.txt) and test set (TREC_10.label.txt),
preprocesses them and then inserts the questions and associated labels into a MySQL database.
'''
import mysql.connector

def data_to_db(TRAIN_DATA_PATH='./data/trec/train_5500.label.txt', TEST_DATA_PATH='./data/trec/TREC_10.label.txt'):
    """
    Load questions and fine category labels to database.
    """
    dataset1 = list(open(TRAIN_DATA_PATH, encoding='utf-8', errors='replace').readlines())
    dataset2 = list(open(TEST_DATA_PATH, encoding='utf-8', errors='replace').readlines())
    dataset = dataset1 + dataset2

    x = [s.split(" ", 1)[1] for s in dataset]
    y = [s.split(' ')[0] for s in dataset]

    broad_classes = {'DESC':'description', 'ENTY':'entity', 'ABBR':'abbreviation','HUM':'human','NUM':'numeric', 'LOC':'location'}
    fine_classes = {'cremat':'creative', 'ind':'individual', 'def':'definition', 'exp':'expression',
                    'dist':'distance', 'lang':'language', 'gr':'group', 'termeq':'term', 'dismed':'medicine', 'mount':'mountain',
                    'tecmeth':'technique', 'volsize':'size', 'instru': 'instrument', 'abb': 'abbreviation', 'perc':'percentage', 'temp': 'temperature',
                    'ord':'order', 'veh':'vehicle', 'desc':'description'}

    y_processed = [] #labels after processing
    for labels in y:
        broad_class, fine_class = labels.split(':')

        broad_class = broad_classes[broad_class]
        if fine_class in fine_classes:
            fine_class = fine_classes[fine_class]

        label = broad_class + " (" + fine_class +")"
        y_processed.append(label)
    return x, y_processed


mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()

sql = "INSERT INTO trec (question, tags) VALUES (%s, %s)"

questions, tags = data_to_db()
max_l = 0
for val in zip(questions, tags):
    if int(len(val[0].strip())) > max_l:
        max_l = len(val[0].strip())
    cursor.execute(sql, val)

mydb.commit()
print(max_l)