import pandas as pd
import mysql.connector
df = pd.read_csv('Law.csv')

mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()
sql = 'insert into compiled (question, tags) values (%s, %s)'

for index, row in df.iterrows():
    row['Tags'] = ""
    if 'criminal' in str(row['Title']).lower():
        row['Tags'] = row['Tags'] + 'criminal-justice' + ", "
    if 'copyright' in str(row['Title']).lower():
        row['Tags'] = row['Tags'] + 'copyright' + ", "
    if 'trademark' in str(row['Title']).lower():
        row['Tags'] = row['Tags'] + 'trademark' + ", "
    if 'constitution' in str(row['Title']).lower():
        row['Tags'] = row['Tags'] + 'constitution' + ", "
    if 'libel' in str(row['Title']).lower():
        row['Tags'] = row['Tags'] + 'libel' + ", "

    row['Tags'] = row['Tags'] + 'legal' + ", "

    print(row['Title'], "--", row['Tags'])

    cursor.execute(sql, (row['Title'], row['Tags']))
    mydb.commit()

