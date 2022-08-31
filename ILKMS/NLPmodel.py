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
from tensorflow.keras.models import load_model 
import os
from streamlit_chat import message as st_message
import time


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents_banglish.json', 'r', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

            
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
key = 0
    
if "history" not in st.session_state:
    st.session_state.history = []
    
#st.title("Hello Chatbot")



#st.session_state.history.append({"message": message, "is_user": True})    

#st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)
col1, col2, col3 = st.columns([3.5, 2, 3])
    
from PIL import Image
image = Image.open('Bangladesh_Logo.png') 
with col2:
    st.image(image, caption=None, channels="RGB", output_format="auto", width= 100)
    
st.markdown("<h5 style='text-align: center; color: black; font-weight: bold;'>Intelligent Land Knowledge Management System <br>for Ministry of Land</h5>", unsafe_allow_html=True)

#with col2:
#st.title("Go! Bot is running!")
#st.subheader("Intelligent Land Knowledge")
#st.subheader("Management System for Ministry of Land")
#messag = st.text_input("Enter your question here...")
#st.write(messag)
container = st.container()
#container.write("Here is the chat history")
message = st.text_input("Enter your question here...", key="input_text")

# st.write(message)
ints = predict_class(message)
    #print(ints)
res = get_response(ints, intents)
#st.session_state.history.append({"message": res, "is_user": False})




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#for background image




footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 5%;
bottom: 0;
width: 90%;
background-color: white;
color: black;
text-align: center;
overflow:auto;

margin:0;
}
.footerP{
color:black;
font-family:FiraSans-Regular, sans-serif;
font-size:12px;
margin:0;
line-height:22.5vh;
}

.socialMedias{
float:right;
font-family:FiraSans-Regular, sans-serif;
font-size:12px;
}
.socialMedias2{
float:left;
font-family:FiraSans-Regular, sans-serif;
font-size:12px
}
.facebook{
width:5vw;
height:10vh;
}
</style>
<div class="footer">
<p style='text-align: right' class = "socialMedias">Design and Developed by <img src="https://imgs.search.brave.com/I5o3znS3QqZZYU2R7W7aP15HHbIIyPCqfGFmtp_TvZ4/rs:fit:706:225:1/g:ce/aHR0cHM6Ly90c2Ux/LmV4cGxpY2l0LmJp/bmcubmV0L3RoP2lk/PU9JUC5IWUtRXzRI/ME8xZ3k4Sko1ME85/aDRBQUFBQSZwaWQ9/QXBp" width = 12% height = 12%></p>
<p style='text-align: left' class = "socialMedias2">Copyrights © 2022, Ministry of Land, All Rights Reserved</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
while message:
    key = key+1
    os.remove('goodd.mp3')
    tts = gTTS(text=res, lang='bn', slow=False)
    tts.save("goodd.mp3")
    user_message = st.session_state.input_text
        #os.system("goodd.mp3")
    #with col2:
    st.audio("goodd.mp3")
    
    st.session_state.history.append({"message": user_message, "is_user": True, "avatar_style": "miniavs"})
    st.session_state.history.append({"message": res, "is_user": False})
    #st.write(res)
    
    with st.expander('', expanded=True):
        for chat in st.session_state.history:
            key = key+1
            st_message(**chat, key = str(key))  # unpacking 
            
    playsound("goodd.mp3")
    break


                           



