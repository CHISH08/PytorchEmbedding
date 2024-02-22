import torch
import torch.nn as nn
import torch.optim as optim


text = ['Сегодня', 'вечером', 'идет', 'сильный', 'дождь', '.']

vocab_size = len(text)
N = 10

context_idxs = torch.tensor([[1, 2, 4, 5]], dtype=torch.long)
target_word = torch.tensor([3], dtype=torch.long)

class CBOW(nn.Module):
    def __init__(self, V, N):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(V, N)
        self.linear = nn.Linear(N, V)

    def forward(self, context_idxs):
        # 1
        idx_embed = self.embeddings(context_idxs)
        # 2
        embeds = torch.sum(idx_embed, dim=1)
        # 3
        out = self.linear(embeds)
        # 4
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

model = CBOW(vocab_size, N)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#7
for epoch in range(201):
    model.zero_grad()
    
    log_probs = model(context_idxs)
    #5
    loss = loss_function(log_probs, target_word)
    
    loss.backward()
    #6
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# После того, как модель обучена, получаем эмбеддинги
embeddings = model.embeddings.weight.to('cpu').data.numpy()

tsne = TSNE(n_components=2, perplexity=min(5, vocab_size-1), random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# Функция для визуализации эмбеддингов
def plot_embeddings(embeddings, labels):
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = embeddings[i]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

word_to_ix = {word: i for i, word in enumerate(text)}
words = list(word_to_ix.keys())

plot_embeddings(embeddings_2d, words)

print(next(iter(model.embeddings.parameters()))[context_idxs[0]])
