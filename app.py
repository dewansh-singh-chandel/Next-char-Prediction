# import torch
# import streamlit as st
# import pickle
# from torch import nn
# from training import NextChar

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model and mappings
# model = NextChar(block_size=30, vocab_size=38, emb_dim=5, hidden_size=500).to(device)
# model.load_state_dict(torch.load("next_char_model.pth", map_location=device))
# with open("stoi.pkl", "rb") as f:
#     stoi = pickle.load(f)
# with open("itos.pkl", "rb") as f:
#     itos = pickle.load(f)

# model.eval()

# g = torch.Generator()
# g.manual_seed(4000002)

# def generate_text(model, itos, stoi, start_str, block_size, max_len=10):
#     context = [0] * block_size
#     text = start_str
#     for i in range(max_len):
#         x = torch.tensor(context).view(1, -1).to(device)
#         y_pred = model(x)
#         ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
#         ch = itos[ix]
#         text += ch
#         context = context[1:] + [ix]
#     return text

# st.title("Next Character Prediction")

# input_text = st.text_input("Enter some text:", value="To be or not to be, that is the question")
# k = st.slider("Number of characters to predict:", min_value=1, max_value=1000, value=20)

# if st.button("Predict"):
#     predicted_text = generate_text(model, itos, stoi, input_text, 30, k)
#     st.write("Predicted text: \n", predicted_text)


import torch
import streamlit as st
import pickle
from torch import nn
from training import NextChar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(model, itos, stoi, start_str, block_size, max_len=10, q=4000002):
    context = [0] * block_size
    text = start_str
    g = torch.Generator()
    g.manual_seed(q)
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        text += ch
        context = context[1:] + [ix]
    return text

st.title("Next Character Prediction")

block_size = st.slider("Block Size:", min_value=10, max_value=20, value=10)
emb_dim = st.slider("Embedding Dimension:", min_value=2, max_value=6, value=4)
q = st.slider("Seed (q):", min_value=0, max_value=10000000, value=4000002)

input_text = st.text_input("Enter some text:", value="To be or not to be, that is the question")
k = st.slider("Number of characters to predict:", min_value=1, max_value=1000, value=20)

if st.button("Predict"):
    model_path = f"next_char_model_{block_size}_{emb_dim}.pth"
    stoi_path = "stoi.pkl"
    itos_path = "itos.pkl"

    model = NextChar(block_size, vocab_size=38, emb_dim=emb_dim, hidden_size=500).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    with open(stoi_path, "rb") as f:
        stoi = pickle.load(f)
    with open(itos_path, "rb") as f:
        itos = pickle.load(f)

    predicted_text = generate_text(model, itos, stoi, input_text, block_size, k, q)
    st.write("Predicted text:", predicted_text)
