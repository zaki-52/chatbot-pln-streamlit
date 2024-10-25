import json
import random

from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers import pipeline

def load_label2id(filename):
    with open(filename) as file:
        config = json.load(file)
        label2id = config['label2id']
    return label2id

def load_intents(filename):
    with open(filename, encoding="utf8") as file:
        intents = json.load(file)
    return intents

def load_model():
    model_path = "./Model/chatbot-exp1"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer= BertTokenizerFast.from_pretrained(model_path)
    chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return chatbot
    
def generate_text(text):
    model = load_model()
    score = model(text)[0]['score']
    label2id = load_label2id(filename='./Model/chatbot-exp1/config.json')
    intents = load_intents(filename='./Model/intents.json')

    if score < 0.8:
        response = "Sorry I can't answer that."
        return response
    
    label = label2id[model(text)[0]['label']]
    response = random.choice(intents['intents'][label]['responses'])
    return response