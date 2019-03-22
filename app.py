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

@app.route('/',methods=['GET','POST'])
def index():
    if request.args.get("dataset"):
        table = request.args.get("dataset")

        with graph.as_default():
            global classifier, model, embeddings_index
            del classifier, model, embeddings_index
            classifier = SentenceClassifier()
            model, embeddings_index = classifier.setup_classifier(table)

    return render_template('index.html')

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
    return render_template('classify.html', question = question, missing_words = missing_words, possible_tags = possible_tags)

@app.route('/question/add/', methods=['GET','POST'])
def add_questions():
    mydb, cursor = classifier.connect_to_db()
    msg = []
    if request.method == 'POST':
        question = str(request.form.get('question'))
        question = classifier.clean_str(question)
        tags = str(request.form.get('tags'))

        question_addable = True #can question and tags be added to db

        if len(question.strip())<10:
            msg.append("Question should be a minimum of 10 characters.")
            question_addable = False
        if len(tags.strip())==0:
            msg.append("No tags were specified.")
            question_addable = False

        if question_addable is True:
            sql = "INSERT INTO " + table + "(question, tags) VALUES (%s, %s)"
            print(question, tags)
            cursor.execute(sql, (question, tags))
            mydb.commit()

            lemmatizer = WordNetLemmatizer()
            #check for word missing in vocab
            for word in question.split(' '):
                if lemmatizer.lemmatize(word) not in embeddings_index:
                    msg.append(word + " not in vocabulary.")
            msg.append("Question added to database.")

    cursor.execute("select tags from " + table)
    tags = list(tag for tag in cursor.fetchall())

    processed_tags = set()

    for ele in tags:
        if ',' not in ele[0]:
            processed_tags.add(ele[0])
        else:
            for tag in ele[0].split(','):
                processed_tags.add(tag.strip())

    return render_template('add.html', embeddings_index = embeddings_index, tags = processed_tags, msg=msg)




if __name__ == '__main__':
    table = 'trec' #Which dataset to use. Default is 'TREC-6'
    graph = tf.get_default_graph()
    classifier = SentenceClassifier()
    model, embeddings_index = classifier.setup_classifier(table)
    app.run(debug=True)
