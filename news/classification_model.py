import re
import joblib
import ssl
import nltk
ssl._create_default_https_context = ssl._create_unverified_context
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import num2words

# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

def find_key_by_value(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)


def predict_article_category(article_text):
    # Load the category code mapping
    category_code_mapping = {'news': 9, 'business': 0, 'health': 7, 'entertainment': 5, 'sport': 13, 
                             'politics': 10, 'culture': 3, 'comedy': 1, 'crime': 2, 'education': 4, 
                             'environment': 6, 'media': 8, 'religion': 11, 'science': 12, 'tech': 14, 'women': 15}

    # Load the saved SVM model
    svm_model = joblib.load('svm_model1.pkl')

    # Load the saved TfidfVectorizer
    tfidf_vectorizer = joblib.load('tf-idf1.pkl')

    # Define cleaning function
    def clean_text(web_text):
        # Lowercasing
        web_text = str(web_text)
        text_clean = web_text.lower()

        # Replace currency symbols with corresponding words
        money_char_mapping = {"$": "dollar", "€": "euro", "£": "pound", "¥": "yen", "₣": "franc", "₹": "rupee"}
        text_clean = "".join([money_char_mapping.get(char, char) for char in text_clean])

        tokens = word_tokenize(text_clean)

        cleaned_tokens = []
        for i in tokens:
            try:
                cleaned_token = num2words.num2words(i, lang='en') if i.isdigit() else i
                cleaned_tokens.append(cleaned_token)
            except Exception as e:
                print(f"Error processing token: {i}, Error: {e}")
                cleaned_tokens.append(i)

        # Convert numeric tokens to words using num2words
        text_clean = " ".join(cleaned_tokens)

        # Remove non-alphabetic characters
        text_clean = re.sub(r'[^a-z]', ' ', text_clean)

        tokens = word_tokenize(text_clean)

        # Remove stop words
        stop_words = set(nltk.corpus.stopwords.words("english"))
        tokens_no_stopwords = [word for word in tokens if word not in stop_words]

        # Lemmatization
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_list = [wordnet_lemmatizer.lemmatize(word) for word in tokens_no_stopwords]

        return " ".join(lemmatized_list)

    # Clean the text
    cleaned_text = clean_text(article_text)

    # Apply TF-IDF transformation
    tfidf_text = tfidf_vectorizer.transform([cleaned_text])

    # Make predictions using the loaded SVM model
    predicted_category_code = svm_model.predict(tfidf_text)[0]

    # Map the predicted category code to the corresponding category
    predicted_category = find_key_by_value(category_code_mapping, predicted_category_code)
    return predicted_category

