import os
import random
import json
import pickle
import numpy as np
from dotenv import load_dotenv

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model

import openai

load_dotenv()

ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()
intents = json.load(open('dataset/intents.json'))

words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            extend = ''
            if "extend" in i:
                extend = " " + i["extend"]
            break
    return result, extend


def answer_generator(quest, ans):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt_gpt = f'parafrase kalimat berikut dengan gaya kasual, seperti sedang menjawab pertanyaan "{quest}?": "{ans}"'

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_gpt,
        temperature=0.8,
        max_tokens=709,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    res_text = response.choices[0].text
    res = res_text[res_text.rindex('\n')+1:]
    return res


print("FisBot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res, extend = get_response(ints, intents)
    res = answer_generator(message, res) + extend
    print("FisBot: " + res)
    print()
