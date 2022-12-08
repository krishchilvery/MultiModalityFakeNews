# Reference - https://github.com/isegura/MultimodalFakeNewsDetection


# Import libraries
import numpy as np # version - 1.21.6
import time
import matplotlib # version - 3.5.3
import matplotlib.image as mpimg
import pandas as pd # version - 1.3.5
from torch import optim # version - 1.11.0
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer # sklearn version - 1.0.2
import nltk # version - 3.7
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
import re # version - 2.2.1
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer # tensorflow version - 2.6.4
import spacy # version - 3.3.1
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F


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


# Stop-words removal and lemmatization
train_stwrd_lem = []
valid_stwrd_lem = []
test_stwrd_lem = []

for new in train_news_clean_1:
    train_stwrd_lem.append(remove_stopwords_lem(new))

for new in valid_news_clean_1:
    valid_stwrd_lem.append(remove_stopwords_lem(new))

for new in test_news_clean_1:
    test_stwrd_lem.append(remove_stopwords_lem(new))

news_all = train_stwrd_lem + valid_stwrd_lem + test_stwrd_lem

tokenizer = Tokenizer(num_words = 128022)
tokenizer.fit_on_texts(news_all)

# Tokenize news

train_tokenized = tokenizer.texts_to_sequences(train_stwrd_lem)
valid_tokenized = tokenizer.texts_to_sequences(valid_stwrd_lem)
test_tokenized = tokenizer.texts_to_sequences(test_stwrd_lem)


# Function to count the lenght of each sequence
def length_squences(data):
    lengths = []
    for i in range(len(data)):
        lengths.append(len(data[i]))
    return lengths


# Pad/truncate the tokenized news

train_tokenized_pad = pad_sequences(train_tokenized, maxlen = 128, truncating = 'pre', padding = 'pre')
valid_tokenized_pad = pad_sequences(valid_tokenized, maxlen = 128, truncating = 'pre', padding = 'pre')
test_tokenized_pad = pad_sequences(test_tokenized, maxlen = 128, truncating = 'pre', padding = 'pre')

# Transform data arrays into tensors

train_tensor = torch.Tensor(train_tokenized_pad).int()
valid_tensor = torch.Tensor(valid_tokenized_pad).int()
test_tensor =  torch.Tensor(test_tokenized_pad).int()

# Tranform tensors into data loader objects

train_set = TensorDataset(train_tensor, torch.Tensor(np.array(train_labels)))
trainloader = DataLoader(train_set, batch_size=64)

valid_set = TensorDataset(valid_tensor, torch.Tensor(np.array(valid_labels)))
validloader = DataLoader(valid_set, batch_size=64)

test_set = TensorDataset(test_tensor, torch.Tensor(np.array(test_labels)))
testloader = DataLoader(test_set, batch_size=64)


# Function to load the word embeddings

def load_embedd(filename):
    words = []
    vectors = []
    file = open(filename,'r', encoding="utf8")
    for line in file.readlines():
       row = line.split(' ')
       vocab = row[0]
       embd = row[1:len(row)]
       embd[-1] = embd[-1].rstrip()
       embd = list(map(float,embd)) # convert string to float
       words.append(vocab)
       vectors.append(embd)
    file.close()
    return words,vectors


# Function to create the embedding matrix

def embed_matx(word_index, vocab, embeddings, length_vocab, length_embedding):
    embedding_matrix = np.zeros((length_vocab +1, length_embedding))
    for word, i in word_index.items():
        if word in vocab:
            idx = vocab.index(word)
            vector =  embeddings[idx]
            embedding_matrix[i] = vector
        if i == length_vocab:
            break
    return embedding_matrix


vocab_gv_300, vectors_gv_300 = load_embedd(filename = "./glove-embeddings/glove.6B.300d.txt") # path to be changed based on the path of data file

word_index = tokenizer.word_index
# Embedding matrix
embedding_matrix_gv_300 = embed_matx(word_index = word_index, vocab = vocab_gv_300, embeddings = vectors_gv_300, 
                             length_vocab = 142720, length_embedding = 300)


# Model
class CNN(nn.Module):
    def __init__(self, nlabels, train_parameters = True, random_embeddings = True): 
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings = 142720, embedding_dim = 300)
        
        if random_embeddings == True:
            self.embedding.weight = nn.Parameter(torch.rand(142720, 300), requires_grad = train_parameters)
        else:
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_matrix_gv_300), requires_grad = train_parameters)
            
        # Filters for the CNN    
        self.filter_sizes = [2,3,4,5]
        self.num_filters = 50
        
        # Concolutional layers
        self.convs_concat = nn.ModuleList([nn.Conv2d(1, self.num_filters, (K, 300)) for K in self.filter_sizes])
        
        # Linear layers
        self.linear1 = nn.Linear(200,128)
  
        self.linear2 = nn.Linear(128,nlabels)
    
        self.relu = nn.ReLU()
        
        self.logsoftmax = nn.LogSoftmax(dim=1) 
        
        
    def forward(self, x):
        # Embedding
        x = self.embedding(x)
        # Unsqueeze
        x = x.unsqueeze(1)
        # Convolution
        x = [F.relu(conv(x.float())).squeeze(3) for conv in self.convs_concat]
        # Max-pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
        x = torch.cat(x, 1)
        # Linear layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.logsoftmax(x) 
        return x


# Function for training
class CNN_extended(CNN):
    
    def __init__(self,nlabels, train_parameters, random_embeddings, epochs=100, lr=0.001):
        
        super().__init__(nlabels, train_parameters, random_embeddings)  
        
        self.lr = lr # Learning Rate
        
        self.optim = optim.Adam(self.parameters(), self.lr)
        
        self.epochs = epochs
        
        self.criterion = nn.NLLLoss()              
        
        # A list to store the loss evolution along training
        
        self.loss_during_training = [] 

        self.valid_loss_during_training = []
        
    def trainloop(self,trainloader, validloader):
        
        
        # Optimization Loop
        
        for e in range(int(self.epochs)):

            start_time = time.time()
            
            # Random data permutation at each epoch
            
            running_loss = 0.
            
            i = 0
            
            length = 0
            
            accuracies = []
            
            for news, labels in trainloader:             
        
                self.optim.zero_grad()  # Reset gradients
            
                out = self.forward(news.int())

                loss = self.criterion(out,labels.long())
                
                loss.backward()

                running_loss += loss.item()

                self.optim.step()
                
                top_p, top_class = out.topk(1, dim=1)
                
                equals = (top_class == labels.view(news.shape[0], 1))
                
                length += news.shape[0]
                
                accuracies.append(sum(equals)) 
                
                accuracy = sum(accuracies)/length
                
                i += 1
                
                if i%1000 == 0:
                    print(" Train accuracy: ", accuracy)
                
            self.loss_during_training.append(running_loss/len(trainloader))

            # Validation Loss
            
            with torch.no_grad():            
                
                running_loss = 0.
                
                i = 0
                
                length = 0
                
                accuracies = []
                
                for news,labels in validloader:
                    
                    out = self.forward(news.int())

                    loss = self.criterion(out,labels.long())

                    running_loss += loss.item()   
                    
                    top_p, top_class = out.topk(1, dim=1)
                
                    equals = (top_class == labels.view(news.shape[0], 1))
                
                    length += news.shape[0]
                
                    accuracies.append(sum(equals)) 
                
                    accuracy = sum(accuracies)/length
                    
                print(" Validation accuracy: ", accuracy)                    
                      
                self.valid_loss_during_training.append(running_loss/len(validloader))

            if(e % 1 == 0): # Every 10 epochs

                print("Training loss after %d epochs: %f" 
                      %(e,self.loss_during_training[-1]), "Validation loss after %d epochs: %f" %(e,self.valid_loss_during_training[-1]),
                      "Time per epoch: %f seconds"%(time.time() - start_time))


# Initialize model
CNN_train_not_random = CNN_extended(nlabels = 2, epochs = 100, lr = 0.003, train_parameters = True, random_embeddings = False)
# Train model
CNN_train_not_random.trainloop(trainloader, validloader)

# We join the train and validation datasets and we train the model with this dataset using the optimal number of epochs. 
# Then we evaluate the model with the test set. First we need to modify a bit the class used previously.

# Join train and validation sequences
train_valid_tokenized_pad = np.concatenate((train_tokenized_pad, valid_tokenized_pad), axis = 0)
# Join train and validation labels
train_valid_labels = np.concatenate((np.array(train_labels), np.array(valid_labels)), axis = 0)

# Create tensor objects

train_valid_tensor = torch.Tensor(train_valid_tokenized_pad).int()

test_tensor =  torch.Tensor(test_tokenized_pad).int()

# Tranform tensors into data loader objects

train_valid_set = TensorDataset(train_valid_tensor, torch.Tensor(np.array(train_valid_labels)))
train_valid_loader = DataLoader(train_valid_set, batch_size=60)

test_set = TensorDataset(test_tensor, torch.Tensor(np.array(test_labels)))
testloader =  DataLoader(test_set, batch_size=60)

# Initialize model
CNN_test_train_not_random = CNN_extended(nlabels = 2, epochs = 100, lr = 0.003, train_parameters = True, random_embeddings = False)
# Train model
CNN_test_train_not_random.trainloop(train_valid_loader)
# Get predictions
predictions_2 = CNN_test_train_not_random.eval_performance(testloader)

print(classification_report(np.array(test_labels).reshape(len(test_labels),1), predictions_2))

# Confusion matrix
print(confusion_matrix(np.array(test_labels).reshape(len(test_labels),1), predictions_2))



     
