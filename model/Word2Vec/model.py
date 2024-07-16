from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from .wv_types import CBOWDataset
from .wv_types import SGDataset
import math

class Word2Vec(nn.Module):
    def __init__(self, dim: int, ws: int, vocab_size: int, sg: bool=False, lr: float=3e-4, test_size: float=0.2, device: str='cpu'):
        super().__init__()
        self.device = device
        self.ws = ws
        self.sg = sg
        self.test_size = test_size

        self.W1 = nn.Embedding(vocab_size, dim)
        self.W1_weight = self.W1.weight
        self.W2 = nn.Embedding(vocab_size, dim)
        self.W2_weight = self.W2.weight

        if sg:
            self.dataset = SGDataset
        else:
            self.dataset = CBOWDataset

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss(reduction='sum')
        self.metric = lambda x: math.exp(x)

        self.to(device)

    def __getitem__(self, word_idx):
        self.eval()
        return self.W1_weight[word_idx].detach().cpu().clone()

    def forward(self, X):
        X = self.W1(X)
        X = torch.mean(X, dim=-2)
        X = torch.mm(X, self.W2_weight.T)
        return X

    def fit(self, tokens, batch_size: int, num_epochs: int=1, num_workers: int=0):
        train_dataloader, test_dataloader, train_size, test_size = self.build_loader(tokens, batch_size, num_workers)

        first_time = time.time()
        train_loss_list = []
        test_loss_list = []
        train_accuracy_list = []
        test_accuracy_list = []

        for epoch in range(1, num_epochs+1):
            train_loss, train_accuracy = self.train_epoch(train_dataloader, train_size)
            test_loss, test_accuracy = self.test_epoch(test_dataloader, test_size)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)

            print(f'Epoch [{epoch}/{num_epochs}, time: {(time.time() - first_time)/60:.3f} minutes]:\n\
                            Train Loss {train_loss:.6f}, Train Perplexity {train_accuracy:.3f}\n\
                            Test Loss {test_loss:.6f}, Test Perplexity {test_accuracy:.3f}')

        return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

    def train_epoch(self, train_dataloader, data_size):
        train_loss = 0
        self.train()

        for context, target in train_dataloader:
            context, target = context.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            probs = self.forward(context)
            loss = self.loss(probs, target.squeeze(-1))
            loss.backward()
            self.optim.step()
            train_loss += loss.item()

        train_loss = train_loss / data_size
        train_accuracy = self.metric(train_loss)

        return train_loss, train_accuracy

    def test_epoch(self, test_dataloader, data_size):
        test_loss = 0
        self.eval()

        with torch.no_grad():
            for context, target in test_dataloader:
                context, target = context.to(self.device), target.to(self.device)
                probs = self.forward(context)
                test_loss += self.loss(probs, target.squeeze(-1)).item()

            test_loss = test_loss / data_size
            test_accuracy = self.metric(test_loss)

        return test_loss, test_accuracy

    def k_Nearest(self, word_idx, k, words_n_grams=None, return_embed=False, use_cosine=False):
        self.eval()
        with torch.no_grad():
            word_vec = self[word_idx].detach().cpu().clone()
            if words_n_grams:
                all_vec = [self[word_n_grams] for word_n_grams in words_n_grams]
                all_vec = torch.stack(all_vec)
            else:
                all_vec = self.W1_weight.detach().cpu().clone()

            all_vecs = all_vec.clone()

            if use_cosine:
                word_vec = word_vec / word_vec.norm(dim=-1, keepdim=True)
                all_vecs = all_vecs / all_vecs.norm(dim=-1, keepdim=True)

                if len(word_vec.shape) == 1:
                    word_vec = word_vec.unsqueeze(0)

                cosine_similarity = torch.mm(all_vecs, word_vec.t()).squeeze()
                distances = 1 - cosine_similarity
            else:
                distances = torch.sqrt(torch.sum(torch.pow(all_vecs - word_vec, 2), dim=-1))

            k_min_dist = torch.argsort(distances)[:k+1]
            k_min_dist_idx = k_min_dist.tolist()

            if return_embed: 
                embed = all_vec[k_min_dist].detach().cpu().clone()
                k_min_dist_idx = (k_min_dist_idx, embed)

        return k_min_dist_idx

    def build_loader(self, tokens, batch_size, num_workers):
        train_tokens = tokens[:round(len(tokens)*(1-self.test_size))]
        test_tokens = tokens[round(len(tokens)*(1-self.test_size)):]

        train_dataset = self.dataset(train_tokens, self.ws)
        test_dataset = self.dataset(test_tokens, self.ws)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_dataloader, test_dataloader, len(train_dataset), len(test_dataset)

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

    @staticmethod
    def plot_embeddings(words: list[str], embeddings):
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)

        x = embeddings_2d[:, 0]
        y = embeddings_2d[:, 1]

        fig = go.Figure()

        for i in range(len(words)):
            fig.add_trace(go.Scatter(
                x=[x[i]],
                y=[y[i]],
                mode='markers+text',
                text=[words[i]],
                textposition='top center',
                marker=dict(size=8, color='rgba(135, 206, 250, 0.6)', line=dict(width=1, color='DarkSlateGrey')),
                name=words[i]
            ))

        fig.update_layout(
            title='Word Embeddings Visualization',
            template='plotly_dark',
            xaxis=dict(title='TSNE Component 1'),
            yaxis=dict(title='TSNE Component 2'),
            height=1000,
            showlegend=True
        )

        fig.show()
