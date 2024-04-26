from model import Word2Vec
from model import FastText
import os

text = ""
with open('voina_i_mir.txt', 'r') as f:
    text = f.read()

# text = "Мама готовила еду, пока папа чистил снег, а их сын и дочь лепили снеговика. \
#         Затем вся семья пошла кушать еду на кухню и смотреть на падающий снег."

embed_dim = 200
window_size = 20
num_epochs = 5
batch_size = 4300
n_gram = 4
nearest_count = 20
device = 'cuda'

cpu_count = os.cpu_count()
tokens = Word2Vec.tokenizer(text)
model1 = FastText(tokens, embed_dim, window_size, n_gram, sg=0, device=device)
res = model1.fit(batch_size, num_epochs, cpu_count)
print(model1.k_Nearest('Андрей', nearest_count, use_cosine=True))
print(model1.k_Nearest('Андрей', nearest_count, use_cosine=False))
print(model1.k_Nearest('война', nearest_count, use_cosine=True))
print(model1.k_Nearest('война', nearest_count, use_cosine=False))
print(model1.k_Nearest('дуб', nearest_count, use_cosine=True))
print(model1.k_Nearest('дуб', nearest_count, use_cosine=False))
print(model1.k_Nearest('Наташа', nearest_count, use_cosine=True))
print(model1.k_Nearest('Наташа', nearest_count, use_cosine=False))

# model2 = Word2Vec(tokens, embed_dim, window_size, sg=0, device=device)
# res = model2.fit(batch_size, num_epochs, cpu_count)
# print(model2.k_Nearest('Андрей', nearest_count, use_cosine=True))
# print(model2.k_Nearest('Андрей', nearest_count, use_cosine=False))
# print(model2.k_Nearest('война', nearest_count, use_cosine=True))
# print(model2.k_Nearest('война', nearest_count, use_cosine=False))
# print(model2.k_Nearest('дуб', nearest_count, use_cosine=True))
# print(model2.k_Nearest('дуб', nearest_count, use_cosine=False))
# print(model2.k_Nearest('Наташа', nearest_count, use_cosine=True))
# print(model2.k_Nearest('Наташа', nearest_count, use_cosine=False))
Word2Vec.plot_metrics(*res)
