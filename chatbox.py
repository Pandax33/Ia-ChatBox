import random
import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents= json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classe = pickle.load(open('classes.pkl','rb'))
model= load_model('charboxModel.model')


def cleanSentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagOfWords(sentence):
    sentenceWords=cleanSentence(sentence)
    bag=[0]*len(words)
    for w in sentenceWords:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def PredictClass(sentence):
    bow= bagOfWords(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    result=[[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]
    result.sort(key=lambda x:x[1],reverse=True)
    returnList=[]
    for r in result :
        returnList.append({'intent':classe[r[0]],'probability':str(r[1])})
    return returnList

def get_Response(intentsList,intentsJson):
    tag= intentsList[0]['intent']
    listOfIntents=intentsJson['intents']
    for i in listOfIntents:
        if i['tag'] == tag:
            result = random.choice(i['reponses'])
            break
    return result

print('Bot is running')

while True:
    message=input("")
    ints=PredictClass(message)
    res=get_Response(ints,intents)
    print(res)