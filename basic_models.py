# Reference - https://github.com/isegura/MultimodalFakeNewsDetection


# Import libraries
import numpy as np # version - 1.21.6
import optuna # version - 3.0.3
import matplotlib # version - 3.5.3
import matplotlib.image as mpimg
import pandas as pd # version - 1.3.5
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # sklearn version - 1.0.2
import nltk # version - 3.7
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re # version - 2.2.1
from tensorflow.keras.preprocessing.text import Tokenizer # tensorflow version - 2.6.4
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

%matplotlib inline

# Train data (with and without images)
traindata_all = pd.read_csv("./all_train.tsv", sep='\t') # path to be changed based on the path of data file
# Validation data (with and without images)
validata_all = pd.read_csv("./all_validate.tsv", sep='\t') # path to be changed based on the path of data file
# Test data (with and without images)
testdata_all = pd.read_csv("./all_test_public.tsv", sep='\t') # path to be changed based on the path of data file

# cleaning the training data
train_data_all = traindata_all[traindata_all['clean_title'].notnull().to_numpy()]
# cleaning the validation data
valid_data_all = validata_all[validata_all['clean_title'].notnull().to_numpy()]
# cleaning the testing data
test_data_all = testdata_all[testdata_all['clean_title'].notnull().to_numpy()]

# separating the data and labels
train_frame = train_data_all["clean_title"]
train_labels = train_data_all["2_way_label"]

valid_frame = valid_data_all["clean_title"]
valid_labels = valid_data_all["2_way_label"]

test_frame = test_data_all["clean_title"]
test_labels = test_data_all["2_way_label"]

# first we remove punctuations and numbers and also multiple spaces

train_list = list(train_frame)
valid_list = list(valid_frame)
test_list = list(test_frame)

train_labels_list = list(train_labels)
valid_labels_list = list(valid_labels)
test_labels_list = list(test_labels)


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


train_news_clean_1 = []
valid_news_clean_1 = []
test_news_clean_1 = []

for new in train_list:
    train_news_clean_1.append(preprocess_text(new))

for new in valid_list:
    valid_news_clean_1.append(preprocess_text(new))

for new in test_list:
    test_news_clean_1.append(preprocess_text(new))


# Function to remove stop words and perform lemmatization
# Initialize lemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# Initialize stemmer and stop_words
#stemmer = PorterStemmer()
stop_words = set(stopwords.words('english')) 


# Function to remove stopwords
def remove_stopwords_lem(text):
    text = word_tokenize(text)
    # Stop words removal
    text = [word for word in text if word not in stop_words]
    # Lematization
    lemmatized_text = []
    for word in text:
        word1 = lemmatizer.lemmatize(word, pos = "n")
        word2 = lemmatizer.lemmatize(word1, pos = "v")
        word3 = lemmatizer.lemmatize(word2, pos = ("a"))
        lemmatized_text.append(word3) 
    text_done = ' '.join(lemmatized_text)
    return text_done


train_lemmatized = [remove_stopwords_lem(text) for text in train_news_clean_1]

valid_lemmatized = [remove_stopwords_lem(text) for text in valid_news_clean_1]

test_lemmatized = [remove_stopwords_lem(text) for text in test_news_clean_1]


# Finding best hyper-parameters for multinomial naive bayes
def objective_Bayes(trial):
    
    # Sample values for the hyper-parameters
    n = trial.suggest_int("n", 1, 2)
    sub_tf = trial.suggest_categorical("sub_tf", ["True", "False"])
    min_df = trial.suggest_int("min_df",5,25)
    # Create pipeline
    Bayes_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1, n), min_df = min_df)),
                            ('tfidf', TfidfTransformer(sublinear_tf = sub_tf)),('classifier', MultinomialNB() )])
    # Fit model
    clf_Bayes = Bayes_pipe.fit(train_lemmatized, train_labels_list)
    # Obtain the predictions
    predictions = Bayes_pipe.predict(valid_lemmatized)
    # Obtain the accuracy
    acc = accuracy_score(valid_labels_list, predictions)
    
    return acc


# Select budget and set seed                            
budget = 40
np.random.seed(0)
# Optimize hyper-parameters
study_Bayes = optuna.create_study(direction="maximize")
study_Bayes.optimize(objective_Bayes, n_trials=budget,show_progress_bar=False)

# Best hyper-parameters
print("Best hyper-parameters: ")
print(study_Bayes.best_params)
# Best score
print("Best score: ")
print(study_Bayes.best_value)


# Finding best hyper-parameters for logistic regression
def objective_Logistic(trial):
    
    # Sample values for the hyper-parameters
    max_iter = trial.suggest_int("max_iter", 320, 420)
    solver = trial.suggest_categorical("solver", ["newton-cg"])
    multi_class = trial.suggest_categorical("multi_class",["ovr", "multinomial"])
    n = trial.suggest_int("n", 1, 2)
    min_df = trial.suggest_int("min_df",5,25)
    sub_tf = trial.suggest_categorical("sub_tf", ["True", "False"])
    # Create pipeline
    Logistic_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1, n), min_df = min_df)),
                            ('tfidf', TfidfTransformer(sublinear_tf = sub_tf)),('classifier', LogisticRegression(random_state = 3,
                                    solver = solver, multi_class = multi_class,   max_iter = max_iter ) )])

    # Fit model
    clf_Logistic = Logistic_pipe.fit(train_lemmatized, train_labels_list)
    # Obtain the predictions
    predictions = Logistic_pipe.predict(valid_lemmatized)
    # Obtain the accuracy
    acc = accuracy_score(valid_labels_list, predictions)
    
    return acc


# Select budget and set seed                            
budget = 40
np.random.seed(0)
# Optimize hyper-parameters
study_Logistic = optuna.create_study(direction="maximize")
study_Logistic.optimize(objective_Logistic, n_trials=budget,show_progress_bar=False)

# Best hyper-parameters
print("Best hyper-parameters: ")
print(study_Logistic.best_params)
# Best score
print("Best score: ")
print(study_Logistic.best_value)


# Finding best hyper-parameters for random forests
def objective_Forest(trial):
    
    # Sample values for the hyper-parameters
    n_estimators = trial.suggest_int("n_estimators", 100, 300)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth = trial.suggest_int("max_depth", 3, 6)
    n = trial.suggest_int("n", 1, 2)
    min_df = trial.suggest_int("min_df",5,25)
    sub_tf = trial.suggest_categorical("sub_tf", ["True", "False"])
    # Create pipeline
    Forest_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1, n), min_df = min_df)),
                            ('tfidf', TfidfTransformer(sublinear_tf = sub_tf)),('classifier', RandomForestClassifier(
                                random_state = 3, n_estimators = n_estimators, criterion = criterion,
                                max_depth = max_depth ) )])

    # Fit model
    clf_Forest = Forest_pipe.fit(train_lemmatized, train_labels_list)
    # Obtain the predictions
    predictions = Forest_pipe.predict(valid_lemmatized)
    # Obtain the accuracy
    acc = accuracy_score(valid_labels_list, predictions)
    
    return acc


# Select budget and set seed                            
budget = 40
np.random.seed(0)
# Optimize hyper-parameters
study_Forest = optuna.create_study(direction="maximize")
study_Forest.optimize(objective_Forest, n_trials=budget,show_progress_bar=False)

# Best hyper-parameters
print("Best hyper-parameters: ")
print(study_Forest.best_params)
# Best score
print("Best score: ")
print(study_Forest.best_value)


# Finding best hyper-parameters for linear SVM
def objective_SVC(trial):
    
    # Sample values for the hyper-parameters
    max_iter = trial.suggest_int("max_iter", 1000, 3000)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    n = trial.suggest_int("n", 1, 2)
    min_df = trial.suggest_int("min_df",5,25)
    sub_tf = trial.suggest_categorical("sub_tf", ["True", "False"])
    # Create pipeline
    SVC_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1, n), min_df = min_df)),
                            ('tfidf', TfidfTransformer(sublinear_tf = sub_tf)),('classifier', LinearSVC(
                                random_state = 3, max_iter = max_iter,
                                loss = loss) )])

    # Fit model
    clf_SVC = SVC_pipe.fit(train_lemmatized, train_labels_list)
    # Obtain the predictions
    predictions = SVC_pipe.predict(valid_lemmatized)
    # Obtain the accuracy
    acc = accuracy_score(valid_labels_list, predictions)
    
    return acc


# Select budget and set seed                            
budget = 40
np.random.seed(0)
# Optimize hyper-parameters
study_SVC = optuna.create_study(direction="maximize")
study_SVC.optimize(objective_SVC, n_trials=budget,show_progress_bar=False)

# Best hyper-parameters
print("Best hyper-parameters: ")
print(study_SVC.best_params)
# Best score
print("Best score: ")
print(study_SVC.best_value)


# Join train and validation sets
training_lemmatized = train_lemmatized + valid_lemmatized

# Joining train and validation labels
training_labels = train_labels_list + valid_labels_list


# Training and testing the multinomial naive bayes classifier
Bayes_pipe_lem = Pipeline([('vect', CountVectorizer(ngram_range = (1, 2), min_df = 5)),
                            ('tfidf', TfidfTransformer(sublinear_tf = 'True')),('classifier', MultinomialNB() )])
Bayes_pipe_lem.fit(training_lemmatized, training_labels)

# Evaluation of the model
predictions_Bayes_lem = Bayes_pipe_lem.predict(test_lemmatized)
print(classification_report(np.array(test_labels).reshape(len(test_labels),1),predictions_Bayes_lem))

# Confusion matrix
print(confusion_matrix(np.array(test_labels).reshape(len(test_labels),1),predictions_Bayes_lem))


# Training and testing logistic regression
Logistic_pipe_lem = Pipeline([('vect', CountVectorizer(ngram_range = (1, 2), min_df = 5)),
                            ('tfidf', TfidfTransformer(sublinear_tf = 'True')),('classifier', LogisticRegression(random_state = 3,
                                    solver = 'newton-cg', multi_class = 'multinomial',   max_iter = 337 ) )])
Logistic_pipe_lem.fit(training_lemmatized, training_labels)

# Evaluation of the model
predictions_Logistic_lem = Logistic_pipe_lem.predict(test_lemmatized)
print(classification_report(np.array(test_labels).reshape(len(test_labels),1),predictions_Logistic_lem))

# Confusion matrix
print(confusion_matrix(np.array(test_labels).reshape(len(test_labels),1),predictions_Logistic_lem))


# Training and testing random forests
Forest_pipe_lem = Pipeline([('vect', CountVectorizer(ngram_range = (1, 2), min_df = 9)),
                            ('tfidf', TfidfTransformer(sublinear_tf = 'True')),('classifier', RandomForestClassifier(
                                random_state = 3, n_estimators = 290, criterion = 'entropy',
                                max_depth = 6) )])
Forest_pipe_lem.fit(training_lemmatized, training_labels)

# Evaluation of the model
predictions_Forest_lem = Forest_pipe_lem.predict(test_lemmatized)
print(classification_report(np.array(test_labels).reshape(len(test_labels),1),predictions_Forest_lem))

# Confusion matrix
print(confusion_matrix(np.array(test_labels).reshape(len(test_labels),1),predictions_Forest_lem))


# Training and testing linear SVM
SVC_pipe_lem = Pipeline([('vect', CountVectorizer(ngram_range = (1, 2), min_df = 5)),
                            ('tfidf', TfidfTransformer(sublinear_tf = 'True')),('classifier', LinearSVC(
                                random_state = 3, max_iter = 1214,
                                loss = 'squared_hinge') )])
SVC_pipe_lem.fit(training_lemmatized, training_labels)

# Evaluation of the model
predictions_SVC_lem = SVC_pipe_lem.predict(test_lemmatized)

# Classification report
print(classification_report(np.array(test_labels).reshape(len(test_labels),1),predictions_SVC_lem))

# Confusion matrix
print(confusion_matrix(np.array(test_labels).reshape(len(test_labels),1),predictions_SVC_lem))






