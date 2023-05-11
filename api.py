from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df = pd.read_csv("OSX_DS_assignment.csv")
df = df[['review_description', 'variety']]
df.dropna(inplace=True)
df['review_description'] = df['review_description'].str.lower()
df['review_description'] = df['review_description'].str.replace('[^\w\s]', '')

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
train_features = vectorizer.fit_transform(df['review_description'])
train_labels = df['variety']

classifier = MultinomialNB()
classifier.fit(train_features, train_labels)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    reviews = data.get('reviews', [])

    new_reviews = pd.Series(reviews)
    new_reviews = new_reviews.str.lower()
    new_reviews = new_reviews.str.replace('[^\w\s]', '')
    new_features = vectorizer.transform(new_reviews)

    predictions = classifier.predict(new_features)

    results = []
    for review, prediction in zip(reviews, predictions):
        results.append({'review': review, 'predicted_wine_variety': prediction})

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run()
