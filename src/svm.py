'''
This file contains the code of the proposed approach which
is based on TF-IDF and SVM methods.
This file can be used to compare the performance of the
proposed approach with the baseline.
'''

########## 1. Import required libraries ##########

import pandas as pd
import numpy as np
import re
import time

# Text and feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation and tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# Classifier : SVM
from sklearn.svm import SVC

# Text cleaning & lemmatizer
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


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

########## 3. Download & read data ##########

# Choose the project (options: 'pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe')
project = 'pytorch'
path = f'datasets/{project}.csv'

pd_all = pd.read_csv(path)
pd_all = pd_all.sample(frac=1, random_state=999)  # Shuffle

# Merge Title and Body into a single column; if Body is NaN, use Title only
pd_all['Title+Body'] = pd_all.apply(
    lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
    axis=1
)

# Keep only necessary columns: id, Number, sentiment, text (merged Title+Body)
pd_tplusb = pd_all.rename(columns={
    "Unnamed: 0": "id",
    "class": "sentiment",
    "Title+Body": "text"
})
pd_tplusb.to_csv('Title+Body.csv', index=False, columns=["id", "Number", "sentiment", "text"])


########## 4. Configure parameters & Start training ##########

# ========== Key Configurations ==========

# 1) Data file to read
datafile = 'Title+Body.csv'

# 2) Number of repeated experiments
REPEAT = 30

# 3) Output CSV file name
out_csv_name = f'results/results.csv'

# ========== Read and clean data ==========
data = pd.read_csv(datafile).fillna('')
text_col = 'text'

# Keep a copy for referencing original data if needed
original_data = data.copy()

# Text cleaning
data[text_col] = data[text_col].apply(remove_html)
data[text_col] = data[text_col].apply(remove_emoji)
data[text_col] = data[text_col].apply(remove_urls)
data[text_col] = data[text_col].replace(to_replace=r'[^\w\s]', value='', regex=True) #remove non-word and non-whitespace characters
data[text_col] = data[text_col].apply(remove_stopwords)
data[text_col] = data[text_col].apply(lemmatize)
data[text_col] = data[text_col].apply(clean_str)


# Lists to store metrics across repeated runs
accuracies = []
precisions = []
recalls = []
f1_scores = []
auc_values = []
time_values = []

for repeated_time in range(REPEAT):

    # start measuring time
    t0 = time.time()

    # --- 4.1 Split into train/test ---
    indices = np.arange(data.shape[0])
    train_index, test_index = train_test_split(
        indices, test_size=0.2, random_state=repeated_time
    )

    train_text = data[text_col].iloc[train_index]
    test_text = data[text_col].iloc[test_index]

    y_train = data['sentiment'].iloc[train_index]
    y_test = data['sentiment'].iloc[test_index]

    # --- 4.2 TF-IDF vectorization ---
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000  # Adjust as needed
    )
    X_train = tfidf.fit_transform(train_text)


    X_test = tfidf.transform(test_text)

    # --- 4.3 SVM Model & Gridsearch ---
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, scoring='f1')

    # fitting the models for grid search
    grid.fit(X_train, y_train)

    # --- 4.4 Make predictions & evaluate ---
    y_pred = grid.predict(X_test.toarray())

    # stop measuring time
    t1 = time.time()

    # Time
    time_value = t1 - t0
    time_values.append(time_value)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Precision (macro)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precisions.append(prec)

    # Recall (macro)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    recalls.append(rec)

    # F1 Score (macro)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_scores.append(f1)

    # AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    auc_val = auc(fpr, tpr)
    auc_values.append(auc_val)


# --- 4.5 Aggregate results ---
scores = []

final_accuracy  = np.mean(accuracies)
scores.append(final_accuracy)

final_precision = np.mean(precisions)
scores.append(final_precision)

final_recall    = np.mean(recalls)
scores.append(final_recall)

final_f1        = np.mean(f1_scores)
scores.append(final_f1)

final_auc       = np.mean(auc_values)
scores.append(final_auc)

avg_score = np.mean(scores)

final_time      = np.mean(time_values)

print(f"=== SVM + TF-IDF Results on {project} project ===")
print(f"Number of repeats:     {REPEAT}")
print(f"Average Accuracy:      {final_accuracy:.4f}")
print(f"Average Precision:     {final_precision:.4f}")
print(f"Average Recall:        {final_recall:.4f}")
print(f"Average F1 score:      {final_f1:.4f}")
print(f"Average AUC:           {final_auc:.4f}")
print(f"Average score:        {avg_score:.4f}")
print(f"Average time:           {final_time:.4f}s")

# Save final results to CSV (append mode)
try:
    # Attempt to check if the file already has a header
    existing_data = pd.read_csv(out_csv_name, nrows=1)
    header_needed = False
except:
    header_needed = True

df_log = pd.DataFrame(
    {
        'Method': "SVM + TF-IDF",
        'Project': project,
        'repeated_times': [REPEAT],
        'Accuracy': [final_accuracy],
        'Precision': [final_precision],
        'Recall': [final_recall],
        'F1': [final_f1],
        'AUC': [final_auc],
        'Average Score': [avg_score],
        'Time': [final_time],
        'CV_list(accuracy)': [str(accuracies)],
        'CV_list(precision)': [str(precisions)],
        'CV_list(recall)': [str(recalls)],
        'CV_list(F1)': [str(f1_scores)],
        'CV_list(AUC)': [str(auc_values)]
    }
)

df_log.to_csv(out_csv_name, mode='a', header=header_needed, index=False)

print(f"\nResults have been saved to: {out_csv_name}")