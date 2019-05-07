import pandas as pd
import re
import mysql.connector


file_tags = ['faq']
mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()

sql1 = 'insert into faq (`question`, `tags`) values (%s, %s)'
sql2 = 'insert into tags (`table`, `broad_label`, `fine_label`) values (%s, %s, %s)'

q_set = []

tag_set = set()
for file in file_tags:

    df = pd.read_csv(file + '.csv')

    for index, row in df.iterrows():
        tags = ""

        tags = tags + row['Tags'].lower()

        print(row['Title'], "--", tags)
        q_set.append([row['Title'],tags])

        tag_list = tags.split(",")
        for tag in tag_list:
            tag_set.add(tag.strip())



    if "" in tag_set:
        tag_set.remove("")
    if " " in tag_set:
        tag_set.remove(" ")
    if "," in tag_set:
        tag_set.remove(",")
    #for tag in tag_set:
        #cursor.execute(sql2, ("faq", tag.lower(), "-"))

    mydb.commit()
cursor.execute(sql1, (row['Title'], tags.lower()))

for i in range(0, len(q_set)):
    cursor.execute(sql1, (q_set[i][0], q_set[i][1]))
print("DONE!!")
mydb.commit()

for i in range(0, len(q_set)-1):
    for j in range(i+1, len(q_set)):
        if "spam" not in q_set[i][1].lower() and "spam" not in q_set[j][1].lower():
            question = q_set[i][0] + " and " + q_set[j][0]
            tag = q_set[i][1] + "," + q_set[j][1]
            cursor.execute(sql1, (question, tag))

            question = q_set[j][0] + " and " + q_set[i][0]
            tag = q_set[j][1] + "," + q_set[i][1]
            cursor.execute(sql1, (question, tag))
    print("DONE!!")
    mydb.commit()


