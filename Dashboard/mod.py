from nltk.corpus import wordnet
from keras.models import load_model
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 
import keras
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
import re
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import contractions
warnings.filterwarnings(action = 'ignore') 
import gensim 
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
from matplotlib import pyplot
from sklearn.decomposition import PCA
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM
from keras.models import Sequential
from keras.layers import MaxPooling1D, Conv1D, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
import chardet
from autocorrect import *
from numpy import loadtxt
from keras.models import load_model
wn=WordNetLemmatizer()
#nlp = spacy.load('en', parse = False, tag=False, entity=False)
tokenizer = Tokenizer()
stop=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
      'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
      'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
      'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
      'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
      'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
      'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
      'the', 'and', 'if', 'or',"'", 'it', ',', 'i', '/','because', 'as',
      'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
      'between', 'into', 'through', 'during', 'above', 'below',
      'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
      'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor',
      'only', 'own', 'same', 'so', 'than', 'too', 's', 't', 'can', 'will',
      'just', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
      'y', 'ain']

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def replace_contractions(text):
    return contractions.fix(text)

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stop]
    return text

def lemmatizer(text):
    text = [wn.lemmatize(word,get_wordnet_pos(word)) for word in text]
    return text

def remove_accented_chars(text):
    x=[]
    for word in text:
        word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        x.append(word)
    return x

def autocorrect(text):
    text = [spell(word) for word in text]
    return text

colnames=["Text","Sentiment"]
df2=pd.read_csv('Tranining_final.csv',encoding="latin-1")
df2=df2.dropna()
df2.columns=colnames

df2["text_new"]=df2["Text"].apply(lambda x: replace_contractions(x))
df2['text_new']=df2['text_new'].str.lower()

df2['text_new'] = df2['text_new'].apply(lambda x: tokenization(x.lower()))
df2["text_new"] = df2["text_new"].apply(lambda x: remove_accented_chars(x))
#df2['text_new'] = df2['text_new'].apply(lambda x: remove_punct(x))
df2['text_new'] = df2['text_new'].apply(lambda x: autocorrect(x))
df2['text_new'] = df2['text_new'].apply(lambda x: remove_stopwords(x))
df2['text_new'] = df2['text_new'].apply(lambda x: lemmatizer(x))
sentences=df2["text_new"].tolist()
model2 =gensim.models.Word2Vec(sentences, min_count = 2,size = 100, window = 5) 
X2 = model2[model2.wv.vocab]
#Initializing PCA with 2 features
pca = PCA(n_components=2)
#Storing the converted 2D results in result 2D array
result = pca.fit_transform(X2)
#Word_vectors
X=[]
for word in model2.wv.vocab:
    X.append(model2.wv[word])

#W containing the universal vocabulary words
W=[]

for word in model2.wv.vocab:
    W.append(word)
    
#Making a word to WV dictionary
word_dict_vec= dict(zip(W, X))
#Function for encoding
def encode_sentence(text):
    
    result = []
    arr = text
    for word in arr:
        w = encode_word(word)
        if w is not None:
            result.append(w)
    return result


def encode_word(word):
    if word not in word_dict:
        return None
    return word_dict[word]

vocab_size = len(model2.wv.vocab)

W_new=sorted(W)
encoded_docs=[]
for i,w in enumerate(W_new):
    encoded_docs.append(i)
    
word_dict={}

keys = W_new
values = encoded_docs

#Dictionary of words to index
word_dict = dict(zip(W_new, values))
#Dictionary of word to vectors
word_dict_vec= dict(zip(W_new, X))

Xts1 = df2["text_new"].values
     
#max_len=50
#Converting sequence to one hot encoding numbers

arr1 = []
for text in Xts1:
    arr1.append(encode_sentence(text))

X1 = keras.preprocessing.sequence.pad_sequences(arr1, maxlen=30)
y1 = df2["Sentiment"].values
X1=np.array(X1)
X1=X1.reshape(X1.shape[0],X1.shape[1])

max_words=30

embedding_size=20
model1=Sequential()
model1.add(Embedding(vocab_size, embedding_size, input_length=max_words))
model1.add(LSTM(100))
model1.add(Dense(1, activation='sigmoid'))
print(model1.summary())

model1.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
import numpy as np
batch_size = 16
num_epochs = 10
np.random.seed(1098)
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.3,random_state=40)
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
model1.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)


scores = model1.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', scores[1])
model1.save("model.h5")

model1 = load_model('model.h5')
df3=pd.read_csv("hollywood_results4.csv")
#sentences3=df3["text_new"].tolist()
#model3 =gensim.models.Word2Vec(sentences3, min_count = 2,size = 50, window = 5) 
df3["dislike_new"]=df3["Dislikes"].apply(lambda x: replace_contractions(x))
df3['dislike_new']=df3['dislike_new'].str.lower()
#df3['dislike_new'] = df3['dislike_new'].apply(lambda x: remove_punct(x))
df3['dislike_new'] = df3['dislike_new'].apply(lambda x: tokenization(x.lower()))
#df3['dislike_new'] = df3['dislike_new'].apply(lambda x: lemmatizer(x))   
#df3["dislike_new"]=df3["dislike_new"].apply(lambda x: remove_accented_chars(x))
df3["dislike_new"]=df3["dislike_new"].apply(lambda x: autocorrect(x))
df3['dislike_new'] = df3['dislike_new'].apply(lambda x: remove_stopwords(x))
df3['dislike_new'] = df3['dislike_new'].apply(lambda x: lemmatizer(x))
df3["like_new"]=df3["Likes"].apply(lambda x: replace_contractions(x))
df3['like_new']=df3['like_new'].str.lower()
#df3['dislike_new'] = df3['dislike_new'].apply(lambda x: remove_punct(x))
df3['like_new'] = df3['like_new'].apply(lambda x: tokenization(x.lower()))
#df3['dislike_new'] = df3['dislike_new'].apply(lambda x: lemmatizer(x))   
#df3["dislike_new"]=df3["dislike_new"].apply(lambda x: remove_accented_chars(x))
df3["like_new"]=df3["like_new"].apply(lambda x: autocorrect(x))
df3['like_new'] = df3['like_new'].apply(lambda x: remove_stopwords(x))
df3['like_new'] = df3['like_new'].apply(lambda x: lemmatizer(x))

Xts_testd = df3["dislike_new"].values

Xts_testl = df3["like_new"].values

     
#max_len=50
#Converting sequence to one hot encoding numbers

arrd = []
arrl = []
for text in Xts_testd:
    arrd.append(encode_sentence(text))
for text in Xts_testl:
    arrl.append(encode_sentence(text))

Xd = keras.preprocessing.sequence.pad_sequences(arrd, maxlen=30)
Xd=Xd.reshape(Xd.shape[0],Xd.shape[1])
yd = model1.predict(Xd)

Xl = keras.preprocessing.sequence.pad_sequences(arrl, maxlen=30)
Xl=Xl.reshape(Xl.shape[0],Xl.shape[1])
yl = model1.predict(Xl)

df4=pd.DataFrame(columns=["Panel","Product","Likes","Likes_new","Sentiment_Likes","Dislikes","Dislikes_new","Sentiment_Dislikes"])   
df4["Likes_new"]=df3["Likes_new"]
df4["Likes"]=df3["Likes"]
df4["Sentiment_Likes"]=yl
df4["Dislikes_new"]=df3["Dislikes_new"]
df4["Product"]=df3["Product"]
df4["Panel"]=df3["Panel"]
#df4.to_csv("hollywood_results3.csv")
df4["Dislikes_new"]=df3["dislike_new"]
df4["Dislikes"]=df3["Dislikes"]
df4["Sentiment_Dislikes"]=yd

products=df4["Product"].unique()
list_pos={}
list_neg={}

  
#df4.to_csv("hollywood_results4.csv")


for product in products:
    list_pos[product]=0
    list_neg[product]=0
for product in products:
    list_pos[product]=df4[(df4["Product"]==product) & (df4["Sentiment_Likes"]>0.4)].count()[1]
    list_neg[product]=df4[(df4["Product"]==product) & (df4["Sentiment_Likes"]<0.4)].count()[1]
