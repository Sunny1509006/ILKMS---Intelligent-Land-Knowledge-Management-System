import random
import json
import pickle
import numpy as np
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gtts import gTTS
from playsound import playsound
import os


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents_banglish.json', 'r', encoding='utf-8').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#print(documents)
    
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

#print(words)

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
    
    
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1 )
model.save('Chatbot_model.model')
#print('Done')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    result.sort(key=lambda x:x[1], reverse=True)
    
    return_list = []
    
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list


def get_response(intents_list, intents_json):
    if intents_list == []:
        print("Sorry, I didn't get you.")
    else:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['response'])
                break
        return result

st.write("Go! Bot is running!")
#messag = st.text_input("Enter your question here...")
#st.write(messag)
message = st.text_input("Enter your question here...")
# st.write(message)
ints = predict_class(message)
    #print(ints)
res = get_response(ints, intents)

while message:
        os.remove('goodd.mp3')
        st.write(res)
        tts = gTTS(text=res, lang='bn', slow=False)
        tts.save("goodd.mp3")

        #os.system("goodd.mp3")
        playsound("goodd.mp3")
        st.audio("goodd.mp3")
        break


        


