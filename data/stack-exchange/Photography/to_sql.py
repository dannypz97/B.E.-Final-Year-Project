import pandas as pd
import mysql.connector
df = pd.read_csv('Photography.csv')

mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="questiondb")
cursor = mydb.cursor()
sql1 = 'insert into Photography (question, tags) values (%s, %s)'
sql2 = 'insert into tags (`table`, `broad_label`, `fine_label`) values (%s, %s, %s)'

#for title
mydict1 ={
    'flash':'flash (photography)',
    'filter':'filter (photography)',
    'nikon':'nikon (photography)',
    'sun':'sun (photography)',
    'camera':'camera (photography)',
    'iso':'iso (photography)',
    'panasonic':'panasonic (photography)',
    'canon':'canon (photography)',
    'sensor':'sensor (photography)',
    'lens':'lens (photography)',
    'battery':'battery (photography)',
    'meter':'metering (photography)',
    'dslr':'dslr (photography)',
    'tripod':'tripod (photography)',
    'zoom':'zoom (photography)',
    'raw':'raw (photography)',
    'jpeg':'jpeg (photography)',
    'jpg':'jpeg (photography)',
    'colour': 'colour (photography)',
    'color':'colour (photography)',
    'sony':'sony (photography)',
    'firmware':'firmware (photography)',
    'backup':'backup (photography)',
    'back up':'backup (photography)',
    'light':'lighting (photography)',
    'bright':'lighting (photography)',
    'night':'night (photography)',
    'day':'day (photography)',
    'exposure':'exposure (photography)',
    'aperture':'aperture (photography)',
    'hdr':'hdr (photography)',
    'print':'printing (photography)',
    'vignette':'vignette (photography),effects (photography)',
    'bokeh':'bokeh (photography),effects (photography)',
    'vintage':'vintage (photography)',
    'gimp':'gimp (photography),software (photography)',
    'photoshop':'photoshop (photography),software (photography)',
    'lightroom':'lightroom (photography),software (photography)',
    'publish':'publishing (photography)',
    'focus':'focus (photography)',
    'portrait':'portrait (photography)',
    'landscape':'landscape (photography)',
    'panorama':'panorama (photography)',
    'on mobile':'mobile camera (photography)',
    'mobile camera':'mobile camera (photography)',
    'android':'android (photography),software (photography)',
    'phone':'mobile camera (photography)',
    'storage':'storage (photography)'
}

#for tags
mydict2 = {
    'camera-basics':'camera basics (photography)',
    'children':'children (photography)',
    'software':'software (photography)',
    'equipment-recom':'equipment recommendation (photography)',
    'camera-recom':'equipment recommendation (photography)',
    'legal': 'legal (photography)',
    'copyright': 'copyright (photography)',
    'sdcard':'storage (photography)',
    'cfcard':'storage (photography)',
    'memory-card':'storage (photography)'
}
for index, row in df.iterrows():
    tags = ""


    for key,value in mydict1.items():
        if key in row['Title'].lower() and (value not in tags.lower()):
            tags = tags + value + ","
    for key,value in mydict2.items():
        if key in row['Tags'].lower() and (value not in tags.lower()):
            tags = tags + value + ","


    print(row['Title'], "--", tags)

    #cursor.execute(sql1, (row['Title'], tags))


#mydb.commit()

tag_set = set()
for val in mydict1.values():
    tags = val.split(',')
    for tag in tags:
        if len(tag)>1:
            tag_set.add(tag)
for val in mydict2.values():
    tags = val.split(',')
    for tag in tags:
        if len(tag)>1:
            tag_set.add(tag)

for tag in tag_set:
    print(tag)
    cursor.execute(sql2, ('photography', 'photography', tag))
mydb.commit()
