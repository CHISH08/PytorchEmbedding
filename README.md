# Word and Text Embeddings

## Введение

Недавно увлекся темой эмбеддингов, и решил подробно изучить методы обучения эмбеддингов слов и текстов с их модификациями. Каждая модель написана собственно-ручно для лучшего понимания, что у них под капотом. Также будет описание за что и как отвечает каждый слой модели и особенности модели.

### Модели для обучения эмбеддингов слов:

- Word2Vec
    1) Skip-Gram
    2) CBOW
- FastText
- GloVe

### Модели для обучения эмбеддингов слов и текста одновременно:

- ELMO
- Tranformers
    1) BERT (Encoder)
    2) GPT (Decoder)
    3) T5 (Encoder + Decoder)

## Модификации класссических методов

У первой тройки реализовал такие методы, как:
1. Negative Sampling
2. Hierarhical Softmax: реализовано на сбалансированном бинарном дереве

## Реализация, описание и тест моделей

### Токенизаторы

1) [Реализация обычного токенизатора](./tokenizer/simple_tokenizer.py)  
2) [Реализация токенизатора, разбивающего на n-gramm-ы (FastText)](./tokenizer/fasttexttokenizer.py)

### Word2Vec

[Общая реализация модели](./model/Word2Vec)

#### CBOW (Continuous Bag of Words)
[Реализация составления датасета для CBOW](./model/Word2Vec/wv_types/CBOW/cbow.py)

#### Skip-Gram
[Реализация составления датасета для Skip-Gram](./model/Word2Vec/wv_types/SkipGram/skipgram.py)

### FastText
[Реализация модели](./model/FastText/fasttext.py)

### GloVe (Global Vectors)
[Реализация модели](./model/GloVe/model.py)

### ELMO (Embeddings from Language Models)
[Реализация модели](./model/ELMO/model.py)

### Transformers
[Реализации](./model/Transformers/)

#### BERT (Bidirectional Encoder Representations from Transformers)
[Реализация модели](./model/Transformers/model.py)

#### GPT (Generative Pre-trained Transformer)
[Реализация модели](./model/Transformers/model.py)

#### T5 (Text-to-Text Transfer Transformer)
[Реализация модели](./model/Transformers/model.py)

## Цель

- Реализация всех методов в одном проекте для полного покружения в мир эмбеддингов
- Тест и изучение каждого из методов с программной, математической, философской точки зрения

## Отличие моего проекта от таких реализаций, как от nltk и тд

- Использование torch, а значит и cuda ядер
- Более читаемый код с точки зрения ООП
- Больше методов для работы с моделями и их изучения
- Описание каждой модели и их особенностей
- Описание на русском языке (возможно, потом добавлю также и на английском)

## Визуализация с помощью plotly

### Метрики
![alt text](present/metrics.png)
### Представление эмбеддингов на плоскости (TSNE)
![alt text](present/image.png)

## Дополнительно

- Написал свой токенизатор со всеми нужными методами
