from flask import Flask, render_template, request, redirect, session
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH = 'model/tfidfvectorizer.pkl'
DATA_PATH = 'drug/drugsComTrain.tsv'

vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/index')
def index():
    if 'user_id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/login_validation', methods=['POST'])
def login_validation():
    username = request.form.get('username')
    password = request.form.get('password')

    session['user_id'] = username
    session['domain'] = password

    if username == "admin@gmail.com" and password == "admin":
        return render_template('home.html')
    else:
        err = "Kindly Enter valid User ID/Password"
        return render_template('login.html', lbl=err)
    
    return ""

@app.route('/age', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('age.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = joblib.load('model.joblib')
        np_arr = floats_string_to_input_arr(text)
        make_picture('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('age.html', href=path)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('predict.html')
    else:
        raw_text = request.form['rawtext']

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_list = [clean_text]

            tfidf_vect = vectorizer.transform(clean_list)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]
            df = pd.read_csv(DATA_PATH, sep='\t')
            top_drugs = top_drugs_extractor(predicted_cond, df)
            
            return render_template('predict.html', raw_text = raw_text, 
                                                   result = predicted_cond, 
                                                   top_drugs = top_drugs)
        else:
            raw_text = "There is no text to select"

def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 4. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 5. lemmitization
    lemmatize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 6. space join words
    return ( ' '.join(lemmatize_words) )

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_list = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_list

def make_picture(training_data_filename, model, new_input_arr, output_file):
    # Plot training data with model
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    heights = data['Height']
    x_new = np.arange(19).reshape((19, 1))
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (Years)', 
                                                                                 'y': 'Height (Inches)'})

    fig.add_trace(
        go.Scatter(x=x_new.reshape(x_new.shape[0]), y=preds, mode='lines', name='Model'))

    if new_input_arr is not False:
        # Plot new predictions
        new_preds = model.predict(new_input_arr)
        fig.add_trace(
        go.Scatter(x=new_input_arr.reshape(new_input_arr.shape[0]), y=new_preds, name='New Outputs', mode='markers', marker=dict(
                color='purple',
                size=20,
                line=dict(
                    color='purple',
                    width=2
                ))))

    fig.write_image(output_file, width=800)
    return fig

def floats_string_to_input_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
        
    floats = [float(x.strip()) for x in floats_str.split(',') if is_float(x)]
    as_np_arr = np.array(floats).reshape(len(floats), 1)
    return as_np_arr

if __name__ == "__main__":
    app.run(debug=True)