from flask import Flask, render_template,request
from actualprediction import *
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')



@app.route('/caption')
def imageCaptionGenerator():
    filePath=request.args.get('image')
    caption= imagepreprocess(filePath)
    return render_template('caption.html',name=caption,image_name=filePath)
    "i'm getting filePath"
    #return harsh(filePath)