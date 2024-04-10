import os
from model import CBOW
import torch
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dim = 210
ws = 10
batch_size = 3000
num_epochs = 1
lr = 1e-3
device = 'cuda'

f = open('./data/voina_i_mir.txt','r')
text = f.read()
f.close()

text = CBOW.tokenizer(text)
model = CBOW(text, dim, ws, lr, device, True)
hist = model.train(text, num_epochs, batch_size, os.cpu_count())

print(model.k_Nearest("война", 5))
print(model.k_Nearest("аполеон", 5))
print(model.k_Nearest("Андрей", 5))
print(model.k_Nearest("Наташа", 5))
print(model.k_Nearest("мир", 5))

print(model.k_Nearest("война", 5, use_cosine=True))
print(model.k_Nearest("аполеон", 5, use_cosine=True))
print(model.k_Nearest("Андрей", 5, use_cosine=True))
print(model.k_Nearest("Наташа", 5, use_cosine=True))
print(model.k_Nearest("мир", 5, use_cosine=True))
model.save_embedding()