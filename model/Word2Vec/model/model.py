import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn
import torch
import time
import re
from model.Vocab import build_vocab
from ..CBOW import CBOWDataset
from ..SkipGram import SGDataset

class Word2Vec(nn.Module):
    def __init__(self, tokens, dim, ws, sg=0, device='cpu'):
        super().__init__()
        self.vocab = build_vocab(tokens)
        self.data = self.text_to_idxs(tokens)
        self.device = device
        self.ws = ws

        self.W1_emb = nn.Embedding(len(self.vocab), dim)
        self.W1 = self.W1_emb.weight
        self.W2_emb = nn.Embedding(len(self.vocab), dim)
        self.W2 = self.W2_emb.weight

        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.optim = torch.optim.Adam(self.parameters(), lr=3e-3)
        if sg:
            self.dataset = SGDataset
        else:
            self.dataset = CBOWDataset
        self.to(device)

    def __getitem__(self, word):
        return self.W1[self.vocab[word.lower()]].detach().cpu().clone()

    def forward(self, x):
        x = self.W1[x]
        x = torch.sum(x, dim=-2)
        x = torch.mm(x, self.W2.T)
        return x

    def fit(self, batch_size, num_epochs, num_workers=1):
        train_dataset = self.dataset(self.data[:(len(self.data) // 5 * 4)], self.ws)
        test_dataset = self.dataset(self.data[(len(self.data) // 5 * 4):], self.ws)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        first_time = time.time()
        train_loss_list = []
        test_loss_list = []
        train_accuracy_list = []
        test_accuracy_list = []

        for epoch in range(1, num_epochs+1):
            train_loss = 0
            train_accuracy = 0.
            test_loss = 0
            test_accuracy = 0.

            self.train()
            for context, target in train_dataloader:
                context, target = context.to(self.device), target.to(self.device)
                self.optim.zero_grad()
                probs = self.forward(context)
                loss = self.loss(probs, target)
                loss.backward()
                self.optim.step()
                train_loss += loss.item()
                train_accuracy += (torch.argmax(probs, dim=1) == target).sum().item()
            train_accuracy = train_accuracy / len(train_dataset) * 100

            self.eval()
            with torch.no_grad():
                for context, target in test_dataloader:
                    context, target = context.to(self.device), target.to(self.device)
                    probs = self.forward(context)
                    test_loss += self.loss(probs, target).item()
                    test_accuracy += (torch.argmax(probs, dim=1) == target).sum().item()
                test_accuracy = test_accuracy / len(test_dataset) * 100
            print(f'Epoch [{epoch}/{num_epochs}, time: {(time.time() - first_time)/60:.3f} minutes]:\n\
                            Train Loss {train_loss:.6f}, Train Accuracy {train_accuracy:.3f}%\n\
                            Test Loss {test_loss:.6f}, Test Accuracy {test_accuracy:.3f}%')

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)

        return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

    def text_to_idxs(self, tokens):
        return torch.tensor([self.vocab[word] for word in tokens])

    def k_Nearest(self, word, k, return_embed=False, use_cosine=False):
        word = self.vocab[word.lower()]
        
        if use_cosine:
            word_vec = self.W1[word].detach().clone()
            word_vec = word_vec / word_vec.norm(dim=-1, keepdim=True)
            all_vecs = self.W1.detach().clone()
            all_vecs = all_vecs / all_vecs.norm(dim=-1, keepdim=True)

            if len(word_vec.shape) == 1:
                word_vec = word_vec.unsqueeze(0)

            cosine_similarity = torch.mm(all_vecs, word_vec.t()).squeeze()
            distances = 1 - cosine_similarity
        else:
            word_vec = self.W1[word].detach().clone()
            all_vecs = self.W1.detach().clone()
            distances = torch.sqrt(torch.sum(torch.pow(all_vecs - word_vec, 2), dim=-1))

        k_min_dist = torch.argsort(distances)[:k+1]
        word_list = list(self.vocab)
        k_min_dist_word = [word_list[idx] for idx in k_min_dist.detach().cpu().tolist()]

        if return_embed:
            embed = self.W1[k_min_dist].detach().cpu().clone()
            return k_min_dist_word, embed

        return k_min_dist_word

    @staticmethod
    def tokenizer(text):
        text = text.lower()
        pattern = r'\w+'
        tokens = re.findall(pattern, text)
        return tokens

    @staticmethod
    def plot_metrics(train_loss, train_acc, test_loss, test_acc):
        epochs = list(range(1, len(train_loss) + 1))

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Loss over Epochs', 'Accuracy over Epochs'))
        fig.add_trace(
            go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train Loss', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=test_loss, mode='lines+markers', name='Test Loss', line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train Accuracy', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=test_acc, mode='lines+markers', name='Test Accuracy', line=dict(color='yellow')),
            row=2, col=1
        )

        fig.update_layout(title_text="Training and Testing Metrics", template='plotly_dark')
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
        fig.show()

    def plot_word(self, words, embeddings=None, method='tsne', perplexity=10, pca_components=50):
        if embeddings is None:
            embeddings = self.W1[self.text_to_idxs(words)].detach().cpu()
        embeddings = embeddings.numpy()

        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(perplexity, len(words)-1), random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, pca_components=min(pca_components, len(words)-1))
        else:
            raise ValueError("Invalid method. Choose between 'tsne' and 'pca'.")

        reduced_embeddings = reducer.fit_transform(embeddings)

        fig = go.Figure()
        for i, word in enumerate(words):
            x, y = reduced_embeddings[i]
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', text=[word], textposition="bottom center", name=word))

        fig.update_layout(title=f'Embedding Visualization ({method.upper()})',
                        xaxis_title='Dimension 1',
                        yaxis_title='Dimension 2',
                        )

        fig.show()
