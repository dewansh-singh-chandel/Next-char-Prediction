# import torch
# import torch.nn.functional as F
# from torch import nn
# import pickle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define NextChar class for the model
# class NextChar(nn.Module):
#     def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
#         super().__init__()
#         self.emb = nn.Embedding(vocab_size, emb_dim)
#         self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
#         self.lin2 = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x):
#         x = self.emb(x)
#         x = x.view(x.shape[0], -1)
#         x = torch.tanh(self.lin1(x))
#         x = self.lin2(x)
#         return x

# # Function to generate training data
# def generate_data(text, block_size=10):
#     inputs, targets = [], []
#     for i in range(0, len(text) - block_size, block_size):
#         inputs.append(text[i:i+block_size])
#         targets.append(text[i+block_size])
#     return inputs, targets

# if __name__ == "__main__":
#     # Read data
#     file_path = "shakespeare_input.txt"
#     with open(file_path, "r", encoding="utf-8") as file:
#         text = file.read()

#     unwanted_chars = {'[', ']', '$','3'}
#     text = ''.join(char for char in text if char not in unwanted_chars)

#     text = text.lower()
#     chars = sorted(list(set(text)))
#     stoi = {ch: i+1 for i, ch in enumerate(chars)}
#     stoi["_"]=0
#     itos = {i: ch for ch, i in stoi.items()}

#     # Convert text to numerical sequence
#     numerical_text = [stoi[ch] for ch in text]

#     block_size = 10
#     emb_dim = 6
#     hidden_size = 500

#     inputs, targets = generate_data(numerical_text, block_size)

#     # Move data to GPU
#     X = torch.tensor(inputs).to(device)
#     Y = torch.tensor(targets).to(device)

#     model = NextChar(block_size, len(stoi), emb_dim, hidden_size).to(device)

#     batch_size = 5000
#     loss_fn = nn.CrossEntropyLoss()
#     opt = torch.optim.AdamW(model.parameters(), lr=0.01)

#     epochs = 500
#     for epoch in range(epochs):
#         batch_losses = []
#         for i in range(0, X.shape[0], batch_size):
#             x = X[i:i+batch_size].to(device)
#             y = Y[i:i+batch_size].to(device)

#             y_pred = model(x)
#             loss = loss_fn(y_pred, y)

#             batch_losses.append(loss.item())
#             loss.backward()
#             opt.step()
#             opt.zero_grad()

#         avg_loss = sum(batch_losses) / len(batch_losses)

#     # Save the model and mappings
#     torch.save(model.state_dict(), "next_char_model.pth")
#     with open("stoi.pkl", "wb") as f:
#         pickle.dump(stoi, f)
#     with open("itos.pkl", "wb") as f:
#         pickle.dump(itos, f)


import torch
import torch.nn.functional as F
from torch import nn
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = torch.tanh(self.lin1(x))
        x = self.lin2(x)
        return x


def generate_data(text, block_size=10):
    inputs, targets = [], []
    for i in range(0, len(text) - block_size, block_size):
        inputs.append(text[i:i+block_size])
        targets.append(text[i+block_size])
    return inputs, targets


def train_model(block_size, emb_dim):
    # Read data
    file_path = "shakespeare_input.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    unwanted_chars = {'[', ']', '$', '3'}
    text = ''.join(char for char in text if char not in unwanted_chars)

    text = text.lower()
    chars = sorted(list(set(text)))
    stoi = {ch: i+1 for i, ch in enumerate(chars)}
    stoi["_"] = 0
    itos = {i: ch for ch, i in stoi.items()}

    numerical_text = [stoi[ch] for ch in text]

    inputs, targets = generate_data(numerical_text, block_size)

    X = torch.tensor(inputs).to(device)
    Y = torch.tensor(targets).to(device)

    model = NextChar(block_size, len(stoi), emb_dim,
                     hidden_size=500).to(device)

    batch_size = 1000
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    epochs = 500
    for epoch in range(epochs):
        batch_losses = []
        for i in range(0, X.shape[0], batch_size):
            x = X[i:i+batch_size].to(device)
            y = Y[i:i+batch_size].to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            batch_losses.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        
    torch.save(model.state_dict(),
               f"next_char_model_{block_size}_{emb_dim}.pth")

    return model, stoi, itos


if __name__ == "__main__":
    for i in range(10, 21):
        for j in range(2, 7):
            print(i, j)
            train_model(i, j)
