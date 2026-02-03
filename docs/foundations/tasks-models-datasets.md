# Machine Learning Tasks, Modalities, and Model Selection for Knowledge Distillation

A foundational guide to understanding task types, data modalities, model architectures, and how to select teacher-student pairs for knowledge distillation.

---

## Table of Contents

1. [The Big Picture: Modalities](#the-big-picture-modalities)
2. [Task Types](#task-types)
   - [Two Paradigms](#the-two-fundamental-paradigms)
   - [Discriminative Methods](#discriminative-methods-a-deep-dive) (Classification, Regression, Ranking, Structured, Retrieval)
   - [Generative Methods](#generative-methods-a-deep-dive) (Autoregressive, Diffusion, VAE, GAN, Flow)
3. [Transformer Architectures](#transformer-architectures)
4. [Classification](#classification) → [NLP Classification](#nlp-classification)
5. [Generation](#generation) → [NLP Generation](#nlp-generation)
6. [Diffusion](#diffusion) → [Audio (StyleTTS2)](#audio-styletts2-text-to-speech), [Image](#image-diffusion)
7. [Datasets Deep Dive](#datasets-deep-dive)
8. [Model Architectures](#model-architectures)
9. [Teacher-Student Selection](#teacher-student-selection)
10. [Recommended Model Pairs](#recommended-model-pairs)

---

## The Big Picture: Modalities

Machine learning operates on different types of data called **modalities**:

| Modality | Data Type | Example Inputs | Example Models |
|----------|-----------|----------------|----------------|
| **Text (NLP)** | Sequences of tokens | Sentences, documents, code | BERT, GPT, Llama |
| **Vision (CV)** | Pixel grids | Images, videos | ResNet, ViT, CLIP |
| **Audio** | Waveforms/spectrograms | Speech, music | Whisper, Wav2Vec |
| **Tabular** | Structured rows/columns | Spreadsheets, databases | XGBoost, TabNet |
| **Time Series** | Sequential measurements | Stock prices, sensors | LSTM, Transformer |
| **Multimodal** | Multiple modalities | Image + caption | CLIP, GPT-4V, Gemini |

### Why This Project Focuses on NLP

**Natural Language Processing (NLP)** = ML applied to human language (text).

This project focuses on NLP because:
1. **Transformers dominate NLP** - Clear teacher-student architectures (BERT → DistilBERT)
2. **Well-defined benchmarks** - GLUE, WMT have standard evaluation
3. **Rich soft-label dynamics** - Text classification has interpretable class probabilities
4. **Practical compression needs** - Deploy language models on edge devices

**But the principles transfer:**
- Knowledge distillation works the same way for vision (ResNet → MobileNet)
- The classification vs. generation distinction applies to images too (ImageNet classification vs. image generation)
- Teacher-student capacity gaps matter regardless of modality

### NLP vs Other Modalities

| Aspect | NLP | Computer Vision | Audio |
|--------|-----|-----------------|-------|
| **Input** | Token sequences | Pixel grids (H×W×C) | Waveforms/Mel spectrograms |
| **Tokenization** | WordPiece, BPE, SentencePiece | Patch embedding (ViT) | Frame-based |
| **Attention** | 1D sequence attention | 2D spatial attention | 1D temporal attention |
| **Classification** | Sentiment, NLI, topic | ImageNet, object detection | Speaker ID, emotion |
| **Generation** | Translation, summarization | Image synthesis, inpainting | Speech synthesis, music |
| **Key architectures** | BERT (encoder), GPT (decoder) | ResNet (CNN), ViT (transformer) | Whisper, Wav2Vec |

---

## Task Types

Across all modalities, tasks fall into two fundamental paradigms:

### The Two Fundamental Paradigms

All machine learning models fall into two broad categories based on **what they model**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MACHINE LEARNING MODELS                               │
├─────────────────────────────────┬───────────────────────────────────────────────┤
│         DISCRIMINATIVE          │                 GENERATIVE                    │
│        P(label | input)         │           P(data) or P(output | input)        │
│                                 │                                               │
│  "Given input, what's the       │  "Learn to produce new data that looks        │
│   probability of each class?"   │   like the training distribution"             │
│                                 │                                               │
│  ┌───────────────────────────┐  │  ┌─────────────────────────────────────────┐  │
│  │ • Classification          │  │  │ Methods:                                │  │
│  │ • Regression              │  │  │                                         │  │
│  │ • Ranking                 │  │  │ • Autoregressive (GPT, Llama)           │  │
│  │ • Structured prediction   │  │  │ • Diffusion (SD, StyleTTS2)             │  │
│  └───────────────────────────┘  │  │ • VAE (Variational Autoencoder)         │  │
│                                 │  │ • GAN (Generative Adversarial)          │  │
│  Examples:                      │  │ • Flow-based (Normalizing Flows)        │  │
│  • BERT (classification head)   │  │ • Energy-based                          │  │
│  • ResNet (ImageNet)            │  └─────────────────────────────────────────┘  │
│  • Logistic regression          │                                               │
│                                 │  Examples:                                    │
│  Output: Fixed set of labels    │  • GPT-4, Llama (text)                        │
│  Loss: Cross-entropy            │  • Stable Diffusion (images)                  │
│  Inference: argmax              │  • StyleTTS2 (speech)                         │
│                                 │                                               │
│                                 │  Output: New data (text, images, audio)       │
│                                 │  Loss: Varies by method                       │
│                                 │  Inference: Sampling                          │
└─────────────────────────────────┴───────────────────────────────────────────────┘
```

### Discriminative vs Generative: The Core Distinction

| Aspect | Discriminative | Generative |
|--------|----------------|------------|
| **What it models** | P(label \| input) — the decision boundary | P(data) or P(output \| input) — the data distribution |
| **Goal** | Classify, distinguish, decide | Create, synthesize, produce |
| **Question answered** | "What category is this?" | "What does data like this look like?" |
| **Output space** | Fixed (N classes, or continuous scalar) | Open-ended (sequences, images, audio) |
| **Training signal** | Labels from dataset | Self-supervision or reconstruction |
| **Examples** | BERT+classifier, ResNet, SVM | GPT, Stable Diffusion, VAE, GAN |

**Key insight:** A model architecture can be used for either paradigm. BERT is an encoder that can be:
- **Discriminative:** Add classification head → P(label | text)
- **Generative:** Use for masked language modeling → P(masked_word | context)

### Discriminative Methods: A Deep Dive

Within discriminative modeling, there are different **task types** based on the output structure:

```
                            DISCRIMINATIVE MODELS
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
   CLASSIFICATION    REGRESSION    RANKING    STRUCTURED      RETRIEVAL
                                              PREDICTION

   P(class|x)        E[y|x]        Score      P(y₁...yₙ|x)   Similarity
   Discrete labels   Continuous    ordering   Sequence of     ranking
                     value                    labels

   Examples:         Examples:     Examples:   Examples:      Examples:
   • Sentiment       • Price pred  • Search    • NER          • Dense retrieval
   • ImageNet        • Age estim   • RecSys    • POS tagging  • Semantic search
   • Spam detect     • Similarity  • IR        • Parsing      • CLIP
```

#### 1. Classification

**Core idea:** Map input to one (or more) discrete categories from a fixed set.

```
Input: "This movie was fantastic!"
       ↓
Model: BERT + Linear(768 → num_classes)
       ↓
Logits: [2.1, -0.5]  (for 2 classes)
       ↓
Softmax: [0.93, 0.07]
       ↓
Output: "positive" (argmax)
```

| Subtype | Output | Loss | Example |
|---------|--------|------|---------|
| **Binary** | 2 classes | BCE or CE | Spam/not spam |
| **Multi-class** | 1 of N classes | Cross-entropy | ImageNet (1000 classes) |
| **Multi-label** | Any subset of N | BCE per label | Movie genres |
| **Hierarchical** | Tree-structured | Hierarchical CE | Product taxonomy |

| Property | Description |
|----------|-------------|
| **Strengths** | Well-understood, easy to evaluate, fast inference |
| **Weaknesses** | Requires labeled data, fixed label set |
| **Best for** | Sentiment, spam, topic, intent, object recognition |
| **Examples** | BERT+classifier, ResNet, ViT |

**Distillation insight:** Classification has rich soft-label dynamics. A teacher's [0.7, 0.2, 0.1] reveals more than hard label [1, 0, 0] — this "dark knowledge" is the core of Hinton's distillation.

#### 2. Regression

**Core idea:** Map input to a continuous value (or vector of values).

```
Input: House features (sqft, bedrooms, location)
       ↓
Model: Neural network
       ↓
Output: $485,000 (continuous)
```

| Subtype | Output | Loss | Example |
|---------|--------|------|---------|
| **Scalar** | Single number | MSE, MAE | House price |
| **Vector** | Multiple values | MSE | Bounding box (x, y, w, h) |
| **Distribution** | Mean + variance | NLL | Uncertainty estimation |

| Property | Description |
|----------|-------------|
| **Strengths** | Captures nuance, natural for continuous quantities |
| **Weaknesses** | Sensitive to outliers, harder to interpret |
| **Best for** | Price prediction, similarity scores, age estimation |
| **Examples** | Linear regression, MLP, transformer encoder + regression head |

**Distillation insight:** Regression distillation uses MSE between teacher and student outputs. Temperature doesn't apply (no softmax).

#### 3. Ranking / Learning to Rank

**Core idea:** Learn to order items by relevance, not just classify them.

```
Query: "best pizza nyc"
       ↓
Items: [Restaurant A, Restaurant B, Restaurant C]
       ↓
Scores: [0.9, 0.3, 0.7]
       ↓
Ranking: A > C > B
```

| Approach | What it optimizes | Loss | Example |
|----------|-------------------|------|---------|
| **Pointwise** | Absolute relevance score | MSE, CE | Rating prediction |
| **Pairwise** | Relative order of pairs | Hinge, RankNet | "A > B" |
| **Listwise** | Entire ranking quality | ListMLE, LambdaRank | NDCG optimization |

| Property | Description |
|----------|-------------|
| **Strengths** | Directly optimizes ranking metrics |
| **Weaknesses** | Complex losses, expensive training |
| **Best for** | Search engines, recommendation, IR |
| **Examples** | LambdaMART, RankNet, BERT for ranking |

#### 4. Structured Prediction

**Core idea:** Output a structured object (sequence, tree, graph) where parts are interdependent.

```
Input: "John lives in New York"
       ↓
Model: BERT + token classification head
       ↓
Output: [B-PER, O, O, B-LOC, I-LOC]  (NER tags)
```

| Subtype | Output structure | Example |
|---------|------------------|---------|
| **Sequence labeling** | Label per token | NER, POS tagging |
| **Sequence-to-sequence** | Output sequence | Translation (but often generative) |
| **Parsing** | Tree structure | Dependency parsing, constituency parsing |
| **Graph prediction** | Graph structure | Scene graphs, molecule structure |

| Property | Description |
|----------|-------------|
| **Strengths** | Captures dependencies between outputs |
| **Weaknesses** | More complex inference (CRF, beam search) |
| **Best for** | NER, POS tagging, parsing, slot filling |
| **Examples** | BERT + CRF, BiLSTM-CRF |

**Distillation insight:** Structured prediction distillation can transfer both token-level logits AND structural knowledge (CRF transition probabilities).

#### 5. Retrieval / Embedding

**Core idea:** Learn representations where similar items are close in embedding space.

```
Query: "How to make pasta"
       ↓
Encoder: BERT → query_embedding (768-dim)
       ↓
Compare: cosine_similarity(query_emb, doc_embs)
       ↓
Retrieve: Top-k most similar documents
```

| Approach | Training | Example |
|----------|----------|---------|
| **Contrastive** | Pull positives close, push negatives apart | SimCLR, CLIP |
| **Triplet** | anchor closer to positive than negative | FaceNet |
| **In-batch negatives** | Use batch samples as negatives | DPR, E5 |

| Property | Description |
|----------|-------------|
| **Strengths** | Scalable (embed once, search fast), flexible |
| **Weaknesses** | Requires good negatives, embedding quality varies |
| **Best for** | Semantic search, RAG, image retrieval |
| **Examples** | DPR, ColBERT, CLIP, E5, BGE |

**Distillation insight:** Embedding distillation aligns student and teacher embedding spaces. Critical for retrieval quality.

### Comparison: Discriminative Methods

| Method | Output | Loss | Inference | Example Task |
|--------|--------|------|-----------|--------------|
| **Classification** | Discrete class | Cross-entropy | argmax | Sentiment, ImageNet |
| **Regression** | Continuous | MSE, MAE | Direct output | Price prediction |
| **Ranking** | Ordered list | Pairwise/Listwise | Sort by score | Search, RecSys |
| **Structured** | Sequence/tree | Token CE + CRF | Viterbi, beam | NER, parsing |
| **Retrieval** | Similarity | Contrastive | kNN search | Semantic search |

### Generative Methods: A Deep Dive

Within generative modeling, there are fundamentally different **methods** for learning and sampling from P(data):

```
                              GENERATIVE MODELS
                                     │
         ┌───────────────┬───────────┼───────────┬───────────────┐
         ▼               ▼           ▼           ▼               ▼
   AUTOREGRESSIVE    DIFFUSION      VAE         GAN         FLOW-BASED

   P(x) = ∏P(xₜ|x<t)  Denoise      Encode to   Adversarial   Invertible
                      iteratively   latent z    game          transforms

   Examples:         Examples:     Examples:   Examples:     Examples:
   • GPT, Llama      • DDPM        • VAE-GAN   • StyleGAN    • Glow
   • WaveNet         • Stable Diff • VQ-VAE    • BigGAN      • RealNVP
   • PixelCNN        • StyleTTS2   • DALL-E 1  • ProGAN      • NICE
```

#### 1. Autoregressive Models

**Core idea:** Factor the joint distribution as a product of conditionals. Generate one element at a time, each conditioned on all previous elements.

```
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)

Generation:
  Step 1: Sample x₁ ~ P(x₁)         → "The"
  Step 2: Sample x₂ ~ P(x₂|x₁)      → "The cat"
  Step 3: Sample x₃ ~ P(x₃|x₁,x₂)   → "The cat sat"
  ...
```

| Property | Description |
|----------|-------------|
| **Strengths** | Exact likelihood, stable training, flexible |
| **Weaknesses** | Sequential (slow), can't parallelize generation |
| **Best for** | Text, code, music (sequential data) |
| **Examples** | GPT, Llama, WaveNet, PixelCNN |

#### 2. Diffusion Models

**Core idea:** Learn to reverse a gradual noising process. Start with pure noise, iteratively denoise to get clean data.

```
Forward (fixed):   x₀ → x₁ → x₂ → ... → xₜ (add noise each step)
Reverse (learned): xₜ → xₜ₋₁ → ... → x₀ (remove noise each step)

Training: Given x₀ and t, predict the noise ε that was added
Sampling: Start at xₜ ~ N(0,I), iteratively denoise
```

| Property | Description |
|----------|-------------|
| **Strengths** | High quality, stable training, mode coverage |
| **Weaknesses** | Slow sampling (many steps), memory intensive |
| **Best for** | Images, audio, video (continuous data) |
| **Examples** | DDPM, Stable Diffusion, DALL-E 2/3, StyleTTS2 |

#### 3. Variational Autoencoders (VAE)

**Core idea:** Learn a compressed latent representation. Encode to latent space, decode back to data space.

```
Encode: x → μ, σ (latent distribution parameters)
Sample: z ~ N(μ, σ)
Decode: z → x̂ (reconstruction)

Loss = Reconstruction + KL(posterior || prior)
```

| Property | Description |
|----------|-------------|
| **Strengths** | Learned latent space, fast sampling, principled |
| **Weaknesses** | Blurry outputs, posterior collapse |
| **Best for** | Representation learning, interpolation |
| **Examples** | VAE, β-VAE, VQ-VAE, DALL-E 1 |

#### 4. Generative Adversarial Networks (GAN)

**Core idea:** Two networks play a game. Generator tries to fool discriminator; discriminator tries to detect fakes.

```
Generator G: z (random noise) → fake image
Discriminator D: image → real/fake probability

Training: G minimizes D's ability to distinguish; D maximizes it
```

| Property | Description |
|----------|-------------|
| **Strengths** | Sharp outputs, fast sampling |
| **Weaknesses** | Training instability, mode collapse |
| **Best for** | High-resolution images, style transfer |
| **Examples** | StyleGAN, BigGAN, CycleGAN, Pix2Pix |

#### 5. Flow-Based Models

**Core idea:** Learn invertible transformations. Map simple distribution (Gaussian) to complex data distribution.

```
z ~ N(0, I)  →  f₁  →  f₂  →  ...  →  fₖ  →  x

Each fᵢ is invertible, so we can compute exact likelihood
```

| Property | Description |
|----------|-------------|
| **Strengths** | Exact likelihood, fast sampling, invertible |
| **Weaknesses** | Architectural constraints, high memory |
| **Best for** | Density estimation, audio synthesis |
| **Examples** | Glow, RealNVP, WaveGlow |

### Comparison: Generative Methods

| Method | Training | Sampling | Quality | Diversity | Likelihood |
|--------|----------|----------|---------|-----------|------------|
| **Autoregressive** | Stable | Slow | High | High | Exact |
| **Diffusion** | Stable | Slow | Very High | High | Approximate |
| **VAE** | Stable | Fast | Medium | High | Lower bound |
| **GAN** | Unstable | Fast | High | Lower | None |
| **Flow** | Stable | Fast | Medium-High | High | Exact |

### Key Terminology Reference

| Term | Category | Meaning | Example |
|------|----------|---------|---------|
| **Discriminative** | Paradigm | Model P(label\|input) — classify inputs | BERT + classifier → "positive" |
| **Generative** | Paradigm | Model P(data) — produce new data | GPT → "The cat sat on the mat" |
| | | | |
| **Classification** | Disc. Method | Map input to discrete class(es) | Sentiment, ImageNet |
| **Regression** | Disc. Method | Map input to continuous value | Price prediction, similarity |
| **Ranking** | Disc. Method | Order items by relevance score | Search, recommendations |
| **Structured Pred.** | Disc. Method | Output sequence/tree of labels | NER, POS tagging, parsing |
| **Retrieval** | Disc. Method | Find similar items via embeddings | Semantic search, RAG |
| | | | |
| **Autoregressive** | Gen. Method | Generate sequentially, each token depends on previous | GPT, Llama, WaveNet |
| **Diffusion** | Gen. Method | Iteratively denoise from random noise | Stable Diffusion, StyleTTS2 |
| **VAE** | Gen. Method | Encode-decode through latent bottleneck | VQ-VAE, DALL-E 1 |
| **GAN** | Gen. Method | Generator vs discriminator adversarial game | StyleGAN, BigGAN |
| **Flow** | Gen. Method | Invertible transforms from simple distribution | Glow, RealNVP |
| | | | |
| **Logits** | Output | Raw model outputs before softmax | [2.1, -0.5, 1.3] |
| **Softmax** | Function | Convert logits to probabilities | [2.1, -0.5, 1.3] → [0.65, 0.05, 0.30] |
| **Cross-entropy** | Loss | Compare predicted probs to true labels | -log(P(correct)) |
| **MSE** | Loss | Mean squared error for regression | (pred - true)² |
| **Contrastive** | Loss | Pull positives close, push negatives apart | SimCLR, CLIP |
| **KL Divergence** | Loss | Measure distribution difference | Distillation soft labels |
| **Temperature** | Technique | Control output distribution sharpness | High T = more uniform |
| **Denoising** | Concept | Remove noise from corrupted data | Core of diffusion models |
| **Iterative** | Concept | Repeat same operation, refine each time | Diffusion sampling |
| **Latent space** | Concept | Compressed representation of data | VAE's z, Stable Diffusion's latent |
| **Embedding** | Concept | Dense vector representation of input | BERT's [CLS], word2vec |
| **Sampling** | Inference | Draw from learned distribution | Top-k, nucleus, ancestral |

---

## Transformer Architectures

The transformer (Vaswani et al., 2017) has three main variants, each suited to different tasks:

### Encoder-Only (Bidirectional)

```
Input:  [CLS] The movie was great [SEP]
         ↓     ↓    ↓    ↓    ↓     ↓
       [=== Full bidirectional attention ===]
         ↓     ↓    ↓    ↓    ↓     ↓
       [CLS] emb  emb  emb  emb  [SEP]
         ↓
       Classification head → "positive"
```

**Key characteristics:**
- Each token attends to ALL other tokens (past AND future)
- [CLS] token aggregates sequence-level representation
- Cannot generate text autoregressively

**Examples:** BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA

**Best for:**
- Text classification (sentiment, topic, intent)
- Natural language inference (MNLI, RTE)
- Named entity recognition
- Extractive question answering
- Semantic similarity

**Why:** Bidirectional context allows the model to understand the full meaning of each token by seeing what comes before AND after it. For classification, you need to understand the whole input before making a decision.

### Decoder-Only (Causal/Autoregressive)

```
Input:  Translate to French: Hello
         ↓         ↓      ↓    ↓
       [=== Causal attention (triangular mask) ===]
         ↓         ↓      ↓    ↓
        emb       emb    emb   emb → next token: "Bonjour"
```

**Key characteristics:**
- Each token only attends to PREVIOUS tokens (causal mask)
- Generates one token at a time, left-to-right
- Can be prompted for any task

**Examples:** GPT-2/3/4, Llama, Mistral, Gemma, Phi

**Best for:**
- Text generation (stories, essays, code)
- Conversational AI (chatbots)
- In-context learning (few-shot prompting)
- Translation (with sufficient scale)
- Summarization

**Why:** Causal attention naturally models "what comes next," making these models excellent generators. With scale, they learn to perform many tasks through prompting.

### Encoder-Decoder (Sequence-to-Sequence)

```
Encoder:  The cat sat on the mat
           ↓   ↓   ↓   ↓  ↓   ↓
         [=== Bidirectional ===]
           ↓   ↓   ↓   ↓  ↓   ↓
          h₁  h₂  h₃  h₄  h₅  h₆
                   ↓
                   ↓ (cross-attention)
                   ↓
Decoder:  Le chat était assis sur le tapis
           ↓   ↓    ↓     ↓    ↓  ↓   ↓
         [=== Causal + cross-attention ===]
```

**Key characteristics:**
- Encoder: Bidirectional understanding of input
- Decoder: Causal generation of output
- Cross-attention: Decoder attends to encoder representations

**Examples:** T5, BART, mT5, mBART, NLLB

**Best for:**
- Machine translation
- Text summarization
- Question answering (abstractive)
- Data-to-text generation

**Why:** The encoder fully understands the source, while the decoder generates fluent output. Cross-attention lets the decoder "look back" at the source while generating.

### Architecture Selection Guide

| Task | Best Architecture | Why |
|------|-------------------|-----|
| Sentiment analysis | Encoder (BERT) | Need full context to judge sentiment |
| Named entity recognition | Encoder (BERT) | Each token needs bidirectional context |
| Machine translation | Encoder-decoder (T5) or large decoder (Llama) | Need to transform one sequence to another |
| Text generation | Decoder (GPT, Llama) | Need to predict "what comes next" |
| Extractive QA | Encoder (BERT) | Find span in context (no generation) |
| Abstractive QA | Encoder-decoder (T5) or decoder (Llama) | Generate novel answer text |
| Conversation | Decoder (Llama, GPT) | Natural dialogue generation |

---

## Classification

### NLP Classification

The NLP classification tasks implemented in this project. The mechanics (softmax, cross-entropy, distillation via soft labels) are identical to classification in other modalities.

#### Natural Language Inference (NLI)

NLI tests a model's ability to determine the logical relationship between two sentences.

**Given:**
- **Premise:** "A man is playing guitar on stage."
- **Hypothesis:** "A musician is performing."

**Task:** Classify as:
- **Entailment:** Hypothesis follows from premise ✓
- **Contradiction:** Hypothesis contradicts premise
- **Neutral:** Neither entailment nor contradiction

**Why NLI matters for distillation:**
- 3-class problem reveals soft-label dynamics better than binary
- Requires semantic reasoning, not just pattern matching
- Soft labels from teacher show "how confident" in each class

#### Paraphrase Detection

Determine if two sentences express the same meaning.

**Example:**
- Sentence 1: "How do I improve my English?"
- Sentence 2: "What's the best way to learn English?"
- **Label:** Duplicate (paraphrase)

**Why it matters:**
- Binary classification with imbalanced classes (~63% negative in QQP)
- Tests robustness to class imbalance
- Real-world application (duplicate detection, search)

#### Sentiment Analysis

Classify text by emotional tone or opinion.

**Example:**
- Text: "This movie was absolutely terrible."
- **Label:** Negative

**Variants:**
- Binary (positive/negative)
- Fine-grained (1-5 stars)
- Aspect-based (sentiment toward specific features)

#### Linguistic Acceptability

Determine if a sentence is grammatically correct.

**Example:**
- Sentence: "The author the book wrote left."
- **Label:** Unacceptable (grammatically incorrect)

**Why it matters:**
- Tests syntactic knowledge separate from semantic
- Uses Matthew's correlation (handles class imbalance)

---

## Generation

### NLP Generation

The NLP generation tasks implemented in this project. Autoregressive decoding and sequence-to-sequence patterns apply similarly to image/audio generation.

#### Machine Translation

Convert text from one language to another.

**Example:**
- Source (English): "The weather is nice today."
- Target (French): "Le temps est beau aujourd'hui."

**Challenges:**
- Vocabulary differences (German compounds, Chinese characters)
- Word order changes (English SVO vs Japanese SOV)
- Morphological complexity (Finnish has 15 cases)
- Idiomatic expressions

**Why it matters for distillation:**
- Tests per-token knowledge transfer across large vocabulary
- Teacher's soft labels reveal translation uncertainties
- Different target languages test generalization

#### Text Summarization

Compress a document while preserving key information.

**Types:**
- **Extractive:** Select important sentences verbatim
- **Abstractive:** Generate novel summary text

**Example:**
- Input: [Long news article about climate summit]
- Output: "World leaders agreed to reduce emissions 50% by 2030."

#### Question Answering

Answer questions based on context or knowledge.

**Types:**
- **Extractive:** Find answer span in passage
- **Abstractive:** Generate answer text
- **Open-domain:** Answer without given context

---

## Diffusion

Diffusion models generate data through iterative denoising. Distillation here focuses on **reducing steps** rather than model size.

### The Diffusion Process

```
Forward (adding noise):
  x₀ (clean) → x₁ → x₂ → ... → x_T (pure noise)

Reverse (denoising):
  x_T (noise) → x_{T-1} → ... → x₀ (clean)

Each step: x_{t-1} = model(x_t, t) + noise
```

**Why many steps?** Each step makes a small correction. Larger steps = larger errors = worse quality.

### Diffusion Distillation Strategies

Unlike classification/generation distillation (reduce parameters), diffusion distillation reduces **iterations**.

| Strategy | Steps Reduced | Quality | Method |
|----------|---------------|---------|--------|
| **Progressive** | 1000 → 500 → 250 → ... → 4 | High | Student matches teacher's 2-step output in 1 step |
| **Consistency** | 1000 → 1-4 | Medium-High | Train to predict same output from any timestep |
| **Adversarial** | 1000 → 4 | High | Discriminator ensures student matches teacher quality |

### Audio Diffusion

#### StyleTTS2 (Text-to-Speech)

StyleTTS2 uses diffusion for prosody/style modeling. Multiple components are distillable:

| Component | Role | Distillation Strategy |
|-----------|------|----------------------|
| **Text Encoder** | Text → embeddings | Layer pruning + KD (6 → 3 layers) |
| **Duration Predictor** | Phoneme durations | MSE regression |
| **Style Diffusion** | Prosody generation | Progressive (100 → 4-8 steps) |
| **HiFi-GAN Vocoder** | Mel → waveform | Pruning + KD |

**Quality vs Speed Trade-off:**

| Diffusion Steps | Speed (RTF) | Quality (MOS) | Use Case |
|-----------------|-------------|---------------|----------|
| 100 | 0.8x | 4.2 | Reference quality |
| 25 | 3.0x | 4.0 | High quality |
| 8 | 9.0x | 3.7 | Real-time, slight degradation |
| 4 | 18.0x | 3.4 | Real-time, noticeable drop |

*RTF = Real-Time Factor (>1 = faster than real-time)*

#### Other Audio Diffusion Models

| Model | Task | Original Steps | Distilled Steps |
|-------|------|----------------|-----------------|
| AudioLDM | Text-to-audio | 200 | 4 (consistency) |
| MusicGen | Music generation | 50 | 8 (progressive) |
| Tortoise TTS | Text-to-speech | 500 | ~50 (progressive) |

### Image Diffusion

| Model | Original Steps | Distilled | Method |
|-------|----------------|-----------|--------|
| Stable Diffusion 1.5 | 50 | 4-8 | LCM (Latent Consistency) |
| SDXL | 50 | 4 | SDXL-Turbo (adversarial) |
| Stable Diffusion 3 | 28 | 4-8 | Progressive |

### Key Insight: Two Types of Diffusion Distillation

| Type | Goal | Bottleneck | Standard KD applies? |
|------|------|------------|---------------------|
| **Step reduction** | Same model, fewer iterations | Inference time | No - need progressive/consistency |
| **Parameter reduction** | Smaller model, same steps | Model size | Yes - standard response/feature KD |

This project will explore both, starting with **step reduction** for StyleTTS2.

---

## Datasets Deep Dive

### GLUE Benchmark

The [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) benchmark is the standard for evaluating NLU models.

| Task | Full Name | Type | Size | Classes | Metric |
|------|-----------|------|------|---------|--------|
| **CoLA** | Corpus of Linguistic Acceptability | Single sentence | 8.5k | 2 | Matthew's corr. |
| **SST-2** | Stanford Sentiment Treebank | Single sentence | 67k | 2 | Accuracy |
| **MRPC** | Microsoft Research Paraphrase | Sentence pair | 3.7k | 2 | F1 / Accuracy |
| **STS-B** | Semantic Textual Similarity | Sentence pair | 5.7k | Regression | Pearson/Spearman |
| **QQP** | Quora Question Pairs | Sentence pair | 364k | 2 | F1 / Accuracy |
| **MNLI** | Multi-Genre NLI | Sentence pair | 393k | 3 | Accuracy |
| **QNLI** | Question NLI | Sentence pair | 105k | 2 | Accuracy |
| **RTE** | Recognizing Textual Entailment | Sentence pair | 2.5k | 2 | Accuracy |
| **WNLI** | Winograd NLI | Sentence pair | 634 | 2 | Accuracy |

### MNLI (Multi-Genre Natural Language Inference)

**Source:** Crowd-sourced sentence pairs from 10 genres

**Genres:**
- Fiction, Government, Telephone, Travel, Letters (matched)
- 9/11, Face-to-face, Verbatim, Slate, OUP (mismatched)

**Why MNLI is important:**
1. **Large scale:** 393k training examples
2. **Multi-genre:** Tests generalization across domains
3. **Three classes:** Entailment, contradiction, neutral
4. **Well-studied:** Extensive baseline comparisons available

**Example:**
```
Premise:   "The Old One always comforted me."
Hypothesis: "I was always comforted by the Old One."
Label:      Entailment
```

### QQP (Quora Question Pairs)

**Source:** Real question pairs from Quora

**Task:** Predict if two questions are semantically equivalent

**Key characteristics:**
- **Class imbalance:** ~63% negative (not duplicate)
- **Real-world data:** Actual user questions
- **Semantic similarity:** Beyond surface-level matching

**Example:**
```
Question 1: "How do I read and find my Kindle books?"
Question 2: "How do I find my books on Kindle?"
Label:       Duplicate
```

### Translation Datasets

#### Europarl

**Source:** European Parliament proceedings (parallel translations)

| Property | Value |
|----------|-------|
| Languages | 21 European languages |
| Size | ~2M sentences per language pair |
| Domain | Parliamentary/political discourse |
| Quality | Professional human translations |

**Supported pairs:** en-de, en-fr, en-es, en-it, en-pt, en-nl, en-da, en-fi, etc.

**Why Europarl:**
- Clean, aligned parallel data
- Consistent domain across languages
- Good for controlled experiments
- Moderate size (not overwhelming for research)

#### WMT (Workshop on Machine Translation)

**Source:** News articles and web crawl data

| Dataset | Pair | Size | Domain |
|---------|------|------|--------|
| WMT14 | en-de | 4.5M | News |
| WMT14 | en-fr | 40M | News |
| WMT14 | en-cs | 15M | News |
| WMT14 | en-ru | 2.1M | News |

**Why WMT:**
- Large-scale training data
- Standard benchmark for translation
- Multiple language pairs
- High-quality test sets

---

## Model Architectures

### BERT (Bidirectional Encoder Representations from Transformers)

**Paper:** Devlin et al., 2018 ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805))

| Variant | Layers | Hidden | Heads | Params |
|---------|--------|--------|-------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

**Pre-training objectives:**
1. **Masked Language Modeling (MLM):** Predict randomly masked tokens
2. **Next Sentence Prediction (NSP):** Predict if sentence B follows A

**Why BERT excels at classification:**
- Bidirectional attention captures full context
- [CLS] token provides sequence-level representation
- Pre-trained on massive text corpus (Wikipedia + BookCorpus)
- Fine-tuning is simple and effective

### DistilBERT

**Paper:** Sanh et al., 2019 ([arXiv:1910.01108](https://arxiv.org/abs/1910.01108))

| Property | BERT-base | DistilBERT |
|----------|-----------|------------|
| Layers | 12 | 6 |
| Hidden size | 768 | 768 |
| Parameters | 110M | 66M (40% smaller) |
| Speed | 1x | 1.6x faster |
| GLUE score | 79.5 | 77.0 (97% retained) |

**Distillation approach (triple loss):**
1. **Distillation loss:** KL divergence between teacher/student softmax
2. **MLM loss:** Masked language modeling (same as BERT)
3. **Cosine loss:** Align hidden state directions

**Why DistilBERT is a good student:**
- Same hidden dimension (no projection needed)
- Half the layers (clear compression target)
- Proven to retain 97% of BERT's capability
- Widely available and well-documented

### RoBERTa (Robustly Optimized BERT Approach)

**Paper:** Liu et al., 2019 ([arXiv:1907.11692](https://arxiv.org/abs/1907.11692))

**Improvements over BERT:**
- Removed NSP objective (found to hurt performance)
- Dynamic masking (different mask each epoch)
- Larger batches (8k vs 256)
- More training data (160GB vs 16GB)
- Longer training (500k steps vs 100k)

| Variant | Params | GLUE |
|---------|--------|------|
| RoBERTa-base | 125M | 86.4 |
| RoBERTa-large | 355M | 88.9 |

**Why use RoBERTa:**
- Stronger baseline than BERT
- Same architecture (compatible distillation)
- DistilRoBERTa available as student

### Llama (Large Language Model Meta AI)

**Paper:** Touvron et al., 2023 ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971))

| Variant | Params | Context | Architecture |
|---------|--------|---------|--------------|
| Llama-3.2-1B | 1B | 128k | Decoder-only |
| Llama-3.2-3B | 3B | 128k | Decoder-only |
| Llama-3.1-8B | 8B | 128k | Decoder-only |
| Llama-3.1-70B | 70B | 128k | Decoder-only |

**Key features:**
- Decoder-only transformer (GPT-style)
- Grouped-query attention (efficiency)
- RoPE positional embeddings
- Pre-trained on 15T+ tokens

**Why Llama for generation:**
- Open weights (research-friendly)
- Strong multilingual capability
- Efficient inference
- Multiple size tiers for distillation

---

## Teacher-Student Selection

### Principles of Model Pairing

#### 1. Capacity Gap

The difference in capacity between teacher and student matters.

| Gap | Teacher → Student | Outcome |
|-----|-------------------|---------|
| Too small | BERT-base → BERT-base | No compression benefit |
| Optimal | BERT-base → DistilBERT | Good knowledge transfer |
| Too large | GPT-4 → tiny model | Student can't absorb knowledge |

**Rule of thumb:** 2-4x parameter reduction is typically effective.

#### 2. Architecture Compatibility

Similar architectures transfer knowledge more effectively.

| Teacher | Student | Compatibility |
|---------|---------|---------------|
| BERT-base | DistilBERT | Excellent (same family) |
| RoBERTa-base | DistilRoBERTa | Excellent (same family) |
| BERT-base | GPT-2 | Poor (encoder → decoder) |
| Llama-3B | Llama-1B | Excellent (same family) |
| T5-base | DistilBERT | Moderate (different paradigms) |

#### 3. Vocabulary Alignment

Teacher and student should share tokenization for logit-level distillation.

| Scenario | Challenge |
|----------|-----------|
| Same tokenizer | Direct logit comparison ✓ |
| Different tokenizer | Need token mapping or feature distillation |
| Different vocab size | Logit dimensions don't match |

#### 4. Task Appropriateness

The teacher should be capable of the target task.

| Task | Good Teacher | Why |
|------|--------------|-----|
| Classification | Fine-tuned BERT | Strong discriminative capability |
| Translation | Llama-3B or mT5 | Trained for multilingual generation |
| NER | BERT with token heads | Token-level classification |

### Compression Ratios

| Ratio | Example | Typical Performance |
|-------|---------|---------------------|
| 1.5x | BERT-base (110M) → 73M | 99% retained |
| 2x | BERT-base (110M) → DistilBERT (66M) | 97% retained |
| 3x | Llama-3B → Llama-1B | 90-95% retained |
| 5x+ | Large → Tiny | 80-90% retained |

---

## Recommended Model Pairs

### Classification Tasks

| Task Type | Teacher | Student | Params | Ratio | Notes |
|-----------|---------|---------|--------|-------|-------|
| **General NLU** | bert-base-uncased | distilbert-base-uncased | 110M → 66M | 1.7x | Standard choice |
| **Robust NLU** | roberta-base | distilroberta-base | 125M → 82M | 1.5x | Stronger baseline |
| **Large→Base** | roberta-large | roberta-base | 355M → 125M | 2.8x | Within-family |
| **Biomedical** | biobert-v1.1 | distilbert-base-uncased | 110M → 66M | 1.7x | Domain transfer |
| **Scientific** | scibert | distilbert-base-uncased | 110M → 66M | 1.7x | Domain transfer |

**Why these pairs work:**
- Same tokenizer (WordPiece for BERT family)
- Same hidden dimension (768) for most pairs
- Well-documented baseline performances
- Pre-trained weights available on HuggingFace

### Generation Tasks

| Task Type | Teacher | Student | Params | Ratio | Notes |
|-----------|---------|---------|--------|-------|-------|
| **Translation** | Llama-3.2-3B | Llama-3.2-1B | 3B → 1B | 3x | Same family |
| **Translation** | mT5-base | mT5-small | 580M → 300M | 1.9x | Encoder-decoder |
| **Translation** | NLLB-3.3B | NLLB-600M | 3.3B → 600M | 5.5x | Specialized MT |
| **Summarization** | Llama-3.2-3B | Llama-3.2-1B | 3B → 1B | 3x | Decoder-only |
| **Code** | CodeLlama-7B | CodeLlama-3B | 7B → 3B | 2.3x | Code-specialized |

**Why these pairs work:**
- Same tokenizer and vocabulary
- Similar architecture (layer structure)
- Demonstrated generation capability
- Natural size gradations

### Diffusion Tasks

| Task Type | Teacher | Student | Distillation Type | Notes |
|-----------|---------|---------|-------------------|-------|
| **TTS (StyleTTS2)** | 100 steps | 4-8 steps | Step reduction | Progressive distillation |
| **Image (SD 1.5)** | 50 steps | 4-8 steps | Step reduction | LCM / consistency |
| **Image (SDXL)** | 50 steps | 4 steps | Step reduction | Adversarial (Turbo) |
| **Audio (AudioLDM)** | 200 steps | 4 steps | Step reduction | Consistency models |

**Why diffusion distillation is different:**
- Same model architecture, different inference budget
- Goal: maintain quality with fewer denoising steps
- Methods: progressive (iterative halving), consistency, adversarial
- No direct logit/feature matching — trajectory matching instead

### Cross-Architecture Considerations

When teacher and student have different architectures:

| Scenario | Solution |
|----------|----------|
| Different hidden sizes | Add projection layer (linear or MLP) |
| Different layer counts | Use layer mapping (uniform, last-n) |
| Different attention heads | Map attention patterns or skip attention distillation |
| Different tokenizers | Use feature distillation at representation level |

---

## Summary: Decision Tree

```
What is your task?
│
├─► Classification (fixed labels)
│   │
│   ├─► English only?
│   │   ├─► Yes → BERT-base → DistilBERT
│   │   └─► No → mBERT → DistilmBERT
│   │
│   └─► Domain-specific?
│       ├─► Biomedical → BioBERT → DistilBERT
│       ├─► Scientific → SciBERT → DistilBERT
│       └─► Finance → FinBERT → DistilBERT
│
├─► Generation (produce text)
│   │
│   ├─► Translation?
│   │   ├─► Single pair → Llama-3B → Llama-1B
│   │   └─► Many pairs → NLLB-3.3B → NLLB-600M
│   │
│   └─► Other generation?
│       └─► Llama-3B → Llama-1B (general purpose)
│
└─► Diffusion (iterative generation)
    │
    ├─► Need fewer steps (same quality)?
    │   ├─► Gradual reduction → Progressive distillation
    │   ├─► Aggressive (1-4 steps) → Consistency models
    │   └─► Quality-critical → Adversarial distillation
    │
    └─► Need smaller model (same steps)?
        └─► Standard KD (response/feature distillation)
```

---

## References

### Classification & Generation
1. **BERT:** Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
2. **DistilBERT:** Sanh et al. "DistilBERT, a distilled version of BERT" (2019) - [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
3. **Hinton Distillation:** Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
4. **GLUE:** Wang et al. "GLUE: A Multi-Task Benchmark" (2018) - [gluebenchmark.com](https://gluebenchmark.com/)
5. **Llama:** Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
6. **RoBERTa:** Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
7. **T5:** Raffel et al. "Exploring Transfer Learning with a Unified Text-to-Text Transformer" (2020)

### Diffusion
8. **Progressive Distillation:** Salimans & Ho "Progressive Distillation for Fast Sampling" (2022)
9. **Consistency Models:** Song et al. "Consistency Models" (2023)
10. **SDXL-Turbo:** Sauer et al. "Adversarial Diffusion Distillation" (2023)
11. **LCM:** Luo et al. "Latent Consistency Models" (2023)
12. **StyleTTS2:** Li et al. "StyleTTS 2: Towards Human-Level TTS" (2023)

### Learning Resources
13. **Transformer Architectures:** [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/6)
14. **GLUE Explained:** [Chris McCormick's Blog](https://mccormickml.com/2019/11/05/GLUE/)
15. **DistilBERT Overview:** [Zilliz Learn](https://zilliz.com/learn/distilbert-distilled-version-of-bert)
