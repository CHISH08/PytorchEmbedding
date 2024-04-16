import os
from model import CBOW

dim = 300
ws = 5
batch_size = 3000
num_epochs = 60
lr = 5e-4
device = 'cuda'

f = open('./data/voina_i_mir.txt','r')
text = f.read()
f.close()

text = CBOW.tokenizer(text)

model = CBOW(text, dim, ws, lr, device, hs=True, bias=True)
hist = model.train(text, num_epochs, batch_size, os.cpu_count())

nearest_num = 20
# print(model.k_Nearest("война", nearest_num))
# print(model.k_Nearest("Наполеон", nearest_num))
# print(model.k_Nearest("Андрей", nearest_num))
# print(model.k_Nearest("Наташа", nearest_num))
# print(model.k_Nearest("мир", nearest_num))

# print(model.k_Nearest("война", nearest_num, use_cosine=True))
# print(model.k_Nearest("Наполеон", nearest_num, use_cosine=True))
# print(model.k_Nearest("Андрей", nearest_num, use_cosine=True))
# print(model.k_Nearest("Наташа", nearest_num, use_cosine=True))
# print(model.k_Nearest("мир", nearest_num, use_cosine=True))
model.save_embedding()

for word in ['война', 'Наполеон', 'Андрей', 'Наташа', 'мир']:
    print(model.k_Nearest(word, nearest_num, use_cosine=True))
    print(model.k_Nearest(word, nearest_num, use_cosine=False))
