from flask import Flask, render_template, request
from classifier import SentenceClassifier
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

app = Flask(__name__)
app.jinja_env.line_statement_prefix = '#' #allows '#' to be used instead of {% %}
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

classifier = None
model = None
embeddings_index = None
table = None           #which table is selected in the beginning
graph = None

@app.route('/',methods=['GET','POST'])
def index():
    global table, classifier, model, embeddings_index, graph

    if classifier is None and model is None and embeddings_index is None and graph is None:
        classifier = SentenceClassifier()
        model, embeddings_index = classifier.setup_classifier(table, load_saved=1)
        graph = tf.get_default_graph()

    if request.args.get("dataset"):
        table = request.args.get("dataset")

        with graph.as_default():
            #global classifier, model, embeddings_index
            classifier = SentenceClassifier()
            model, embeddings_index = classifier.setup_classifier(table)
            graph = tf.get_default_graph()
    return render_template('index.html', table=table)

@app.route('/classify/', methods=['GET', 'POST'])
def classify():
    missing_words = [] #store list of words in question but not in vocab
    question = str(request.form.get('question'))
    words = classifier.clean_str(question)
    lemmatizer = WordNetLemmatizer()

    for word in words.split(' '):
        if lemmatizer.lemmatize(word) not in embeddings_index:
            missing_words.append(word)

    with graph.as_default():
        possible_tags = classifier.tag_question(model, question)
    #print(possible_tags)
    return render_template('classify.html', question = question, missing_words = missing_words, possible_tags=possible_tags)



@app.route('/data/')
def view_dataset():

    mydb, cursor = classifier.connect_to_db()
    print(table)
    label = request.args.get("label")
    if label is None:
        cursor.execute('select distinct(broad_label) from tags where `table` = "' + table + '"')
        result = cursor.fetchall()
    else:
        result = [(label)]

    rows = []

    for row in result:
        if label is None:
            cursor.execute('select question, tags from ' + table + ' where tags like "%' + row[0] + '%" limit 1000')
        else:
            cursor.execute('select question, tags from ' + table + ' where tags like "%' + row + '%" ')
        for element in cursor.fetchall():
            rows.append(element)

    del mydb, cursor
    return render_template('data.html', rows = rows)


if __name__ == '__main__':
    table = 'trec'
    app.run(debug=True, threaded=True)
