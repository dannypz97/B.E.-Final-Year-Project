import pandas as pd
import re
import mysql.connector


file_tags = ['Academia', 'Personal Finance', 'Workplace']
mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()

sql1 = 'insert into compiled (`question`, `tags`) values (%s, %s)'
#sql2 = 'insert into tags (`table`, `broad_label`, `fine_label`) values (%s, %s, %s)'

tag_set = set()
for file in file_tags:

    df = pd.read_csv(file + '.csv')

    for index, row in df.iterrows():
        tags = file.lower()+ ","


        string = re.sub(r">", " (" + file.lower() +"),", row['Tags'])
        string = re.sub(r"<", "", string)
        string = re.sub(r",,", ",", string)
        string = string.strip()
        tags = tags + string

        #print(row['Title'], "--", tags)
        cursor.execute(sql1, (row['Title'], tags.lower()))

        tag_list = tags.split(",")
        for tag in tag_list:
            tag_set.add(tag.strip())

    mydb.commit()

    if "" in tag_set:
        tag_set.remove("")
    if " " in tag_set:
        tag_set.remove(" ")
    if "," in tag_set:
        tag_set.remove(",")
    #for tag in tag_set:
        #cursor.execute(sql2, ("compiled", file.lower(), tag.lower()))

    mydb.commit()



