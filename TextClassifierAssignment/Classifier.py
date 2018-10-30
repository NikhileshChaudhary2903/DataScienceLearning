
#                                         ## Medical Text Classification 
# In[22]:
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer
import re, string, unicodedata, inflect
import contractions,nltk



def remove_non_ascii(words):
    newwords = []
    for w in words:
        new_word = unicodedata.normalize('NFKD', w).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        newwords.append(new_word)
    return newwords

def to_lowercase(words):
    newwords = []
    for w in words:
        new_word = w.lower()
        newwords.append(new_word)
    return newwords

def remove_punctuation(words):
    newwords = []
    for w in words:
        newword = re.sub(r'[^\w\s]', '', w)
        if newword != '':
            newwords.append(newword)
    return newwords

def replace_numbers(words):
    p = inflect.engine()
    newwords = []
    for w in words:
        if word.isdigit():
            new_word = p.number_to_words(w)
            newwords.append(new_word)
        else:
            newwords.append(w)
    return newwords

def remove_stopwords(words):
    newwords = []
    for w in words:
        if w not in stopwords.words('english'):
            newwords.append(w)
    return newwords

def stem_words(words):
    stemmer = LancasterStemmer()
    stems = []
    for w in words:
        stem = stemmer.stem(w)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in words:
        lemma = lemmatizer.lemmatize(w, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    ##words = remove_punctuation(words)
    ##words = replace_numbers(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas

## read in the training set and split into the class label and medical text abstract
## Step 1
def readTrainingSet():
    # open training set and read its lines
    with open("../train.dat", "r") as trs:
         trset = trs.readlines()
        
    # open test set and read its lines    
    with open("../test.dat", "r") as tst:
         testset = tst.readlines()
    
    train_cls_label = [x.split("\t", 1)[0] for x in trset]
    train_text_abstract = [x.split("\t", 1)[1] for x in trset]
    
    return train_cls_label,train_text_abstract,testset
    
## Step 1
train_cls_label,train_text_abstract,testset=readTrainingSet()    

# print(len(train_cls_label))
# print(len(train_text_abstract))
# print(len(testset))

def clean(text_abstracts):
    
    clean_train_abs = []

    for index, abstr in enumerate(text_abstracts):
 
        clean_train_abs.append(preProcess(abstr))
    
    return clean_train_abs

def preProcess(rawAbstract):

    def strip_html(text):
       soup = BeautifulSoup(text)
       return soup.get_text()


    def remove_between_square_brackets(text):
       return re.sub('\[[^]]*\]', '', text)

    def denoise_text(text):
        text = strip_html(text)
        ##text = remove_between_square_brackets(text)
        return text
    ##Noise Removal
    sample = denoise_text(rawAbstract)

    sample = re.sub(r'([\w\.-]+@[\w\.-]+\.\w+)','',sample)
    
    sample = re.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]| \
        [a-z0-9.\-]+[.][a-z]{2,4}/|[a-z0-9.\-]+[.][a-z])(?:[^\s()<>]+|\(([^\s()<>]+| \
        (\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))','', sample)


    def replace_contractions(text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    ##sample = replace_contractions(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    
    stops = set(stopwords.words("english"))                  
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = ''
    for word in words:
        if word not in stops and len(word) > 3:
            lemmatized_words += str(lemmatizer.lemmatize(word)) + ' '
    
    # stemmer = LancasterStemmer()
    # stemmer_words = ''
    # for word in words:
    #     if word not in stops and len(word) > 3:
    #         stemmer_words += str(stemmer.stem(word)) + ' '

    return lemmatized_words    
def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )
# In[26]:
# In[27]:

def createTFIDFMatrices(train_data, test_data):
    
    vectorizer = TfidfVectorizer(norm = 'l2')  
    train_matrix = vectorizer.fit_transform(train_data)
    test_matrix = vectorizer.transform(test_data)

    return train_matrix, test_matrix

def findSimilarities(train_matrix, test_matrix):

    cosineSimilarities = np.dot(test_matrix, np.transpose(train_matrix))
    similarities = cosineSimilarities.toarray()
        
    return similarities

def findKNearest(similarity_vector, k):

    return np.argsort(-similarity_vector)[:k]    

def predict(nearestNeighbors, labels):

    class_labels=[labels[neighbour] for neighbour in nearestNeighbors]
    cls_counter=Counter(class_labels)
    ##print(cls_counter.most_common(1))
    return cls_counter.most_common(1)[0][0]

    

# print(train_text_abstract[0])
##print(testset[0])

train_text_abstract=clean(train_text_abstract)
testset=clean(testset)

# print(train_text_abstract[0])
# print(testset[0])

train_matrix, test_matrix = createTFIDFMatrices(train_text_abstract, testset)
csr_info(train_matrix)
csr_info(test_matrix)

similarities = findSimilarities(train_matrix, test_matrix)
# print(similarities.shape)
# print(type(similarities))
k=100
test_classes=list()

for s in similarities:
    knn = findKNearest(s, k)
    prediction = predict(knn, train_cls_label)
    ##print(type(prediction))
    
    test_classes.append(int(prediction))
  
output = open('output-k.dat', 'w')
output.writelines( "%s\n" % item for item in test_classes)
output.close()

