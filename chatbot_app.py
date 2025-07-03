# === FILE: chatbot_buah_app.py ===
# Chatbot Penjual Buah dengan Streamlit

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import json
import re
import random

# Inisialisasi
stemmer = PorterStemmer()
nltk.download('punkt')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Load model
data = torch.load("data_buah.pth")

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load dataset buah
buah_df = pd.read_csv("dataset_buah.csv")  # Kolom: nama, harga, asal, stok

# Generator respon
def get_response(intent, entity):
    if intent == "cek_harga" and entity.get("nama_buah"):
        result = buah_df[buah_df["nama"].str.lower() == entity["nama_buah"].lower()]
        if not result.empty:
            return f"Harga {result.iloc[0]['nama']} adalah Rp{result.iloc[0]['harga']}/kg."
        else:
            return "Maaf, buah tersebut tidak tersedia saat ini."

    elif intent == "cek_stok" and entity.get("nama_buah"):
        result = buah_df[buah_df["nama"].str.lower() == entity["nama_buah"].lower()]
        if not result.empty:
            return f"Stok {result.iloc[0]['nama']} tersedia sebanyak {result.iloc[0]['stok']} kg."
        else:
            return "Maaf, kami tidak memiliki stok buah itu."

    elif intent == "asal_buah" and entity.get("nama_buah"):
        result = buah_df[buah_df["nama"].str.lower() == entity["nama_buah"].lower()]
        if not result.empty:
            return f"{result.iloc[0]['nama']} berasal dari {result.iloc[0]['asal']}."
        else:
            return "Kami belum tahu asal buah itu."

    elif intent == "daftar_buah":
        buah_list = buah_df["nama"].tolist()
        return "Berikut adalah buah yang tersedia:\n- " + "\n- ".join(buah_list)

    else:
        return "Maaf, saya belum bisa membantu untuk pertanyaan itu."

# Ekstraksi entitas
def extract_entity(text):
    entity = {}
    nama_buah_list = buah_df["nama"].dropna().unique()

    for nama in nama_buah_list:
        if nama.lower() in text.lower():
            entity["nama_buah"] = nama
            break

    return entity

# Prediksi intent
def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        return tag
    else:
        return "unknown"

# Streamlit UI
st.title("🍎 Chatbot Penjual Buah")
st.markdown("Tanyakan tentang harga, stok, atau asal buah yang tersedia.")

user_input = st.text_input("Tanyakan sesuatu:", "Berapa harga apel?")

if st.button("Tanya"):
    intent = predict_class(user_input)
    entity = extract_entity(user_input)
    response = get_response(intent, entity)
    st.write(response)
