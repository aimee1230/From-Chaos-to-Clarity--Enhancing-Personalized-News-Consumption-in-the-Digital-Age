from flask import Flask, request, jsonify, render_template
import pickle
from transformers import  AutoTokenizer
import torch
import re
from classification_model import predict_article_category
from transformers import AutoModelForSeq2SeqLM

app = Flask(__name__, static_url_path='/static', template_folder='templates')

# Define route for the root URL
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify_and_summarize', methods=['POST'])
def classify_and_summarize():
    app.logger.info('Received POST request to /classify_and_summarize')
    app.logger.info('Request Headers: %s', request.headers)
    app.logger.info('Request Data: %s', request.data)
    app.logger.info('Form Data: %s', request.form)
    news_article = request.form.get('news_article')
    app.logger.info('News Article: %s', news_article)

    # Classify news article
    class_prediction = predict_article_category(news_article)

    # Summarize news article (assuming summarization_model has a summarize method)
    model = AutoModelForSeq2SeqLM.from_pretrained("bart")
    tokenizer = AutoTokenizer.from_pretrained("bart")
    token = tokenizer(news_article, return_tensors='pt')
    with torch.no_grad():
        summary_ids =model.generate(
        token['input_ids'],
        num_beams=4,
        max_length=250,
        early_stopping=True
        )
    # decode summary
    news_article_summary = tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
    )
     
    # Return classification result and summary to frontend
    return jsonify({
        'class': class_prediction,
        'summary': news_article_summary
    })

if __name__ == '__main__':
    app.run(debug=True)
