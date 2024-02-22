import os
import sys
sys.path.append('/home/denis/code/Word2Vec/vocab')
from CBOW import CBOW
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
text = ""
f = open('voina_i_mir.txt','r')
text = f.read()
f.close()
text = CBOW.tokenizer(text)
def plot_embeddings(embeddings, labels, word, path):
    plt.figure(figsize=(15, 15))
    plt.title(f"{len(labels)} ближайших слов к слову: {word}")
    for i, label in enumerate(labels):
        x, y = embeddings[i]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(f"{path}/words/{word}.png")
    plt.close()
vocab_size = 20
tsne = PCA(n_components=2)
words = [
    "война", "мир", "любовь", "аристократия", "судьба", "героизм", "семья", "честь", "дружба",
    "революция", "Россия", "Наполеон", "брак", "измена", "жертва", "дуб", "небо",
    "Пьер", "Безухов", "Наташа", "Ростов", "Андрей", "Болконский", "Николай", "Марья", "Курагин", "Москва",
    "поле", "имение", "Лев", "Толстой", "Бонапарт"
]
if not os.path.isdir("train"):
    os.mkdir("train")
    with open('train/result.csv', 'a') as file:
        file.write('word,embedding,window_size,embedding_size,word1,word2,word3,word4,word5,word6,word7,word8,word9,word10,word11,word12,word13,word14,word15,word16,word17,word18,word19,word20\n')
for windows_size in range(81, 102, 20):
    if not os.path.isdir(f"train/ws{windows_size}"):
        os.mkdir(f"train/ws{windows_size}")
    for i, embedding_size in enumerate(range(10, 311, 50)):
        if not os.path.isdir(f"train/ws{windows_size}/emb{embedding_size}"):
            os.mkdir(f"train/ws{windows_size}/emb{embedding_size}")
        if not os.path.isdir(f"train/ws{windows_size}/emb{embedding_size}/words"):
            os.mkdir(f"train/ws{windows_size}/emb{embedding_size}/words")
        model = CBOW(text, embedding_size=embedding_size, windows_size=windows_size, lr=1e-2, num_epochs=35, batch_size=8000 - i * 900, device='cuda', num_workers=os.cpu_count(), log=f"train/ws{windows_size}/emb{embedding_size}")
        for word in words:
            k_word, embed = model.euclid_dist(word, vocab_size)
            embeddings_2d = tsne.fit_transform(embed.tolist())
            with open('train/result.csv', 'a') as file:
                file.write(f"{word},{model[word].tolist()},{windows_size},{embedding_size}," + ",".join(k_word)+"\n")
            plot_embeddings(embeddings_2d, k_word, word, f"train/ws{windows_size}/emb{embedding_size}")
        del model
        torch.cuda.empty_cache()