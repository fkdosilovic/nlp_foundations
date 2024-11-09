# NLP Foundations

Collection of the most important books and (seminal) papers for the modern NLP.

## Contents

- [NLP Foundations](#nlp-foundations)
  - [Contents](#contents)
  - [Books](#books)
  - [Embeddings](#embeddings)
    - [Word embeddings](#word-embeddings)
    - [Sentence and paragraph embeddings](#sentence-and-paragraph-embeddings)
  - [Deep Learning](#deep-learning)
  - [Attention](#attention)
  - [Language Models](#language-models)
    - [Encoder-based models (BERT)](#encoder-based-models-bert)
    - [Decoder-based models (GPT)](#decoder-based-models-gpt)
    - [Encoder-decoder models](#encoder-decoder-models)
    - [Training methodology](#training-methodology)
  - [Large Language Models](#large-language-models)
  - [Efficient Transformers](#efficient-transformers)
    - [Pruning and quantization](#pruning-and-quantization)
    - [Adapters](#adapters)
    - [Low-rank adaptations](#low-rank-adaptations)
    - [Small (Large) Language Models](#small-large-language-models)
  - [Other](#other)

## Books

1. [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) [^1]
2. [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) [^2]

[^1]: Excellent introduction to NLP.
[^2]: Excellent, in-depth introduction to modern decoder-based large language models.

## Embeddings

### Word embeddings

1. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (word2vec)
2. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (word2vec)
3. [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)
4. [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper_files/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)
5. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
6. [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
7. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) (ELMo)
8. [Learning Word Vectors for 157 Languages](https://arxiv.org/abs/1802.06893)

### Sentence and paragraph embeddings

1. [Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053)
2. [From Word Embeddings To Document Distances](https://proceedings.mlr.press/v37/kusnerb15.html)
3. [Siamese CBOW: Optimizing Word Embeddings for Sentence Representations](https://arxiv.org/abs/1606.04640)
4. [Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features](https://arxiv.org/abs/1703.02507)
5. [An efficient framework for learning sentence representations](https://arxiv.org/abs/1803.02893)
6. [Fuzzy Bag-of-Words Model for Document Representation](https://ieeexplore.ieee.org/document/7891009)
7. [Word Mover's Embedding: From Word2Vec to Document Embedding](https://arxiv.org/abs/1811.01713)

## Deep Learning

1. [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078)
2. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
3. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
4. [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023)
5. [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101)
6. [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
7. [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
8. [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)

## Attention

1. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473)
2. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformers)

## Language Models

1. [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
2. [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)

### Encoder-based models (BERT)

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)
3. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
4. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
5. [DistilBERT, A distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
6. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)
7. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)
8. [MobileBERT: A Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)
9. [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
10. [MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers](https://arxiv.org/abs/2012.15828)
11. [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)

### Decoder-based models (GPT)

1. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2)
3. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
4. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT)

### Encoder-decoder models

1. [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
2. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html) (T5)
3. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (FLAN-T5)

### Training methodology

1. [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) (XLM)
2. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
3. [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529)
4. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
5. [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) (XLM-RoBERTa)

## Large Language Models

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
3. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## Efficient Transformers

1. [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

### Pruning and quantization

1. [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
2. [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)
3. [What Matters in Transformers? Not All Attention is Needed](https://arxiv.org/abs/2406.15786)

### Adapters

TODO

### Low-rank adaptations

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
3. [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
4. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)

### Small (Large) Language Models

1. [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
2. [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)

## Other

1. [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)
2. [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)
