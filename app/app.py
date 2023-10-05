from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model = joblib.load('model.joblib')
        np_arr = floats_string_to_input_arr(text)
        make_picture('AgesAndHeights.pkl', model, np_arr, path)

        return render_template('index.html', href=path)
    
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
