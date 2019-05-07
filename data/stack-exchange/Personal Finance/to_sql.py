import pandas as pd
import mysql.connector
df = pd.read_csv('Personal Finance.csv')

mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()
sql1 = 'insert into test (question, tags) values (%s, %s)'
sql2 = 'insert into tags (`table`, `broad_label`, `fine_label`) values (%s, %s, %s)'

for index, row in df.iterrows():
    tags = ""
    if 'stock' in row['Title'].lower():
        tags = tags + "stock (personal finance),"


    print(row['Title'], "--", tags)

    cursor.execute(sql1, (row['Title'], tags))


mydb.commit()

tag_set = set()

