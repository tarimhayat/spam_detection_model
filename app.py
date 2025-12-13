from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import pickle

app = Flask(__name__)

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()   # lower case

    text = nltk.word_tokenize(text)   # tokenize

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)    # removing special characters

    text = y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:   # removing stop words and punctuations
            y.append(i)
            
    text = y.copy()
    y.clear()
    for i in text:               # Stemming
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("Model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    output= ""
    if request.method=="POST":
        text = request.form["input_text"]
        transformed_sms = transform_text(text)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        print(result)
        if result == 1:
            output= "Spam"
        else:
            output= "Ham"
    return render_template("index.html", detect=output)

if __name__ == "__main__":
    app.run(debug=True)
