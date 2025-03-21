########## 1. Import required libraries and load models ##########

import pandas as pd
import re
import joblib
import sys

# Text cleaning & lemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Trained Vectorizer
tfidf = joblib.load('model/vectorizer.pkl')

# Trained Classifier
classifier = joblib.load('model/classifier.pkl')


########## 2. Define text preprocessing methods ##########

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+')
    return url_pattern.sub('', text)

# Stopwords
# The list is the same as NLTK's stop word list but without the 'no' stopwords such as 'no', 'not', 'should' etc
stop_words_list = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'd', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had','has','have','having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's", 'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is', 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just', 'll', 'm', 'ma', 'me', 'more', 'most', 'my', 'myself', 'need', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'we', "we'd", "we'll", "we're", 'were', "we've", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'y', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've", '...']

def remove_stopwords(text):
    """Remove stopwords from the text."""
    return " ".join([word for word in str(text).split() if word not in stop_words_list])

# Lemmatizer
def lemmatize(text):
    """ Lemmatize each word in the text."""
    lem = WordNetLemmatizer()
    return " ".join([lem.lemmatize(word) for word in str(text).split() if len(word)>1])

def clean_str(string):
    """
    Clean text by removing non-alphanumeric characters,
    and convert it to lowercase.
    """
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


########## 3. Read the text file and classify it ##########

if len(sys.argv) < 2:
    print("\n/!\ Please provide the path of the text file you want to classify as the first argument in the command line. /!\ \n")

else :
    try:
        path = sys.argv[1]
        with open(path, 'r',encoding='UTF-8') as file:
            data = file.read()

        # Text cleaning
        data = remove_html(data)
        data = remove_emoji(data)
        data = remove_urls(data)
        data = remove_stopwords(data)
        data = lemmatize(data)
        data = clean_str(data)

        # Text encoding
        report = tfidf.transform([data, ''])

        # Prediction of the class
        predicted_class = classifier.predict(report)[0]

        print("\n----------------------------------------------------------------------------------")
        print("                       --- CLASSIFICATION RESULT ---\n")
        if predicted_class :
            print("POSITIVE: The provided bug report is classified as a performance bug-related.\n")
        else:
            print("NEGATIVE: The provided bug report is not classified as a performance bug-related.\n")

        print("\n----------------------------------------------------------------------------------\n")

    except FileNotFoundError: print("\n/!\ The file you provided was not found. /!\ \n")

