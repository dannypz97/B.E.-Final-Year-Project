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

    if request.args.get("dataset"):
        table = request.args.get("dataset")

    classifier = SentenceClassifier()
    model, embeddings_index = classifier.setup_classifier(table, load_saved=1)
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



if __name__ == '__main__':
    table = 'trec'

    app.run(debug=True, threaded=True)
