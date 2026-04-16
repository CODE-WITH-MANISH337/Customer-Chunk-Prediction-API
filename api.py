from fastapi import FastAPI
from fastapi.responses import HTMLResponse,FileResponse
from pydantic import BaseModel,Field
from typing import Annotated,Optional
import pandas as pd
import numpy as np
import re 
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


ps=PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tf.pkl', 'rb') as f:
    tf = pickle.load(f) 

def remove_tags(text):
    return re.sub(r'<.*?>', '', str(text))
def remove_url(text):
    return re.sub(r'http\S+|www\S+', '', str(text))
def preprocessing(text):
    words = text.lower().split()
    return " ".join(ps.stem(word) for word in words if word not in stop_words)



class InputData(BaseModel):
    message:Annotated[Optional[str],Field(...,description='Give a message for chunk prediction')]


app=FastAPI()    

@app.get('/')
def home():
    return {'message':'Hello this is API of customer chunk prediction for more detail'}


app = FastAPI()

@app.get("/")
def processing():
    # content="5; url=/docs" means wait 5 seconds, then go to /docs
    html_content = """
    <html>
        <head>
            <meta http-equiv="refresh" content="5; url=/about" />
        </head>
        <body>
            <h1>HELLO USER</h1>
            <p>You will be redirected to the dashboard in 5 seconds lets go .......</p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get('/about')
def about():
    return  FileResponse('hello.html')

@app.post('/predict')
def predict(data:InputData):
    input_text = data.message

    input_text=remove_tags(input_text)
    input_text=remove_url(input_text)
    input_text=preprocessing(input_text)

    input_text = tf.transform([input_text])
    prediction = int(model.predict(input_text)[0])
    if prediction==1:
        return {'prediction':'Postive'}
    else:
        return {'prediction':'Negative'}
