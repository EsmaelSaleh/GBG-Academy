# Transformers Explained: Encoder & Decoder Architecture

*A complete, simple-language guide with math, examples, and practical guidance*

---

## 1. What Is a Transformer?

A Transformer is a neural network architecture designed to process sequences (like sentences) **all at once**, instead of one word at a time like older models (RNNs, LSTMs). It was introduced in the 2017 paper *"Attention Is All You Need"*.

**The core idea:** every word in a sentence should be able to "look at" every other word directly, and decide how much attention to pay to each one. This is called **self-attention**.

Think of reading the sentence:

> "The animal didn't cross the street because **it** was too tired."

To understand what "it" refers to, your brain instantly glances back at "animal." A Transformer does the same thing mathematically — it lets every word compute a weighted relationship to every other word.

### Why Transformers replaced RNNs
- RNNs read word-by-word, so they're slow (can't parallelize) and forget long-range information.
- Transformers process the whole sequence in parallel, and self-attention gives them a direct path between any two words, no matter how far apart — no "forgetting" over distance.

---

## 2. The Big Picture: Two Halves

The original Transformer has two main blocks:

```
INPUT SENTENCE            OUTPUT SENTENCE
     │                          │
     ▼                          ▼
 ┌─────────┐              ┌─────────┐
 │ ENCODER │ ── context ─▶│ DECODER │──▶ prediction
 └─────────┘              └─────────┘
```

- **Encoder**: reads the input and builds a rich, contextual understanding of it (a set of numbers/vectors representing meaning).
- **Decoder**: uses that understanding to generate an output, one piece at a time (e.g., translating to another language, or continuing text).

Both are built from the **same basic ingredients** — attention, feed-forward layers, normalization — just arranged differently. Let's build those ingredients first, then assemble the encoder and decoder.

---

## 3. The Core Ingredients (with math)

### 3.1 Tokenization and Embeddings

A sentence is first split into tokens (words or sub-words). Each token is converted into a vector of numbers (an **embedding**) — think of it as the word's "address" in a meaning-space with, say, 512 dimensions.

If the sentence has tokens $x_1, x_2, ..., x_n$, each becomes a vector $e_1, e_2, ..., e_n \in \mathbb{R}^{d}$ (commonly $d = 512$ or $768$).

**Simple analogy:** imagine every word gets a GPS coordinate in a huge space, where similar-meaning words end up near each other (e.g., "king" and "queen" are close, "king" and "banana" are far apart).

### 3.2 Positional Encoding

Problem: unlike RNNs, a Transformer has no built-in sense of word *order* — it sees the whole sentence at once, like a bag of words. So we must inject position information.

The original paper uses sine and cosine functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where:
- `pos` = position of the word in the sentence (0, 1, 2, ...)
- `i` = the dimension index
- `d` = embedding size

**In plain words:** each position gets a unique "wave pattern" fingerprint added to its embedding, so the model can tell "word 1" from "word 5" — and can also infer relative distances between words, since these waves have consistent mathematical relationships.

Final input to the model:
```
input_vector = embedding + positional_encoding
```

### 3.3 Self-Attention — the heart of the Transformer

For each word, the model creates **three vectors** by multiplying the input embedding by three learned weight matrices:

```
Query (Q) = X · W_Q
Key   (K) = X · W_K
Value (V) = X · W_V
```

Where `X` is the matrix of input embeddings, and `W_Q, W_K, W_V` are weight matrices learned during training.

**Intuition:**
- **Query** = "what am I looking for?"
- **Key** = "what do I contain / offer?"
- **Value** = "what information do I actually give if picked?"

Think of a library search: your Query is your search term, every book has a Key (its label/summary), and if a Key matches well, you get that book's Value (its content).

**The attention formula:**

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

Step by step, in plain language:
1. `Q · Kᵀ` — compare every query against every key using a dot product → gives a raw "compatibility score" between every pair of words.
2. `/ √d_k` — scale down the scores (d_k = dimension of the key vectors) so numbers don't get too large and destabilize training.
3. `softmax(...)` — turn the scores into probabilities that sum to 1 across each row (so each word distributes 100% of its "attention" among all words).
4. `· V` — use those probabilities as weights to combine the Value vectors → produces a new vector for each word that blends in relevant context from other words.

**Worked mini example:**

Sentence: "I love cats" (3 words, imagine embeddings of size 2 for simplicity).

Say after projection we get (toy numbers):

| Word | Q | K | V |
|------|-----|-----|-----|
| I | [1,0] | [1,0] | [0,2] |
| love | [0,1] | [0,1] | [1,1] |
| cats | [1,1] | [1,0] | [2,0] |

For the word **"love"**, compute its dot product with every Key:
- love·I = [0,1]·[1,0] = 0
- love·love = [0,1]·[0,1] = 1
- love·cats = [0,1]·[1,0] = 0

Scale (divide by √d_k, here d_k=2, √2≈1.41): scores ≈ [0, 0.71, 0]

Softmax turns these into probabilities ≈ [0.28, 0.44, 0.28] (roughly — softmax exaggerates the highest score).

New vector for "love" = 0.28×V(I) + 0.44×V(love) + 0.28×V(cats)
= 0.28×[0,2] + 0.44×[1,1] + 0.28×[2,0]
= [0.56+0.44+0.56 , 0+0.44+0]  ≈ **[1.56, 0.44]**

So "love"'s new representation is a *blend* of all three words' Values, weighted by relevance — this is self-attention in action.

### 3.4 Multi-Head Attention

Instead of doing this once, the Transformer does it **multiple times in parallel** with different learned weight matrices — called "heads." Each head can learn a different kind of relationship (e.g., one head tracks grammar, another tracks meaning/topic).

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

**Analogy:** imagine 8 different readers each re-reading the sentence looking for a different type of pattern (grammar, tone, references, sentiment...), then combining their notes.

### 3.5 Feed-Forward Network (FFN)

After attention, each word's vector passes through a small neural network (applied identically to every position):

```
FFN(x) = max(0, x·W1 + b1) · W2 + b2
```

This is just two linear layers with a ReLU activation in between. It adds non-linearity and lets the model process each word's blended representation further.

### 3.6 Residual Connections + Layer Normalization

Every sub-layer (attention or FFN) is wrapped like this:

```
output = LayerNorm(x + Sublayer(x))
```

- **Residual connection** (`x + ...`): adds the original input back in, so gradients can flow easily during training (prevents vanishing gradients in deep stacks).
- **LayerNorm**: rescales values so numbers don't blow up or shrink to zero across the network, stabilizing training.

**Analogy:** it's like taking notes on a document while keeping the original next to you — you always have the raw text plus your annotations, never losing the original signal.

---

## 4. The Encoder — Understanding the Input

The Encoder's job: read the full input and produce a context-rich representation of it. It's used in the original Transformer for the *source* sentence, and stands alone in models like BERT.

### Structure of one Encoder layer:

```
Input Embeddings + Positional Encoding
        │
        ▼
 ┌───────────────────────┐
 │ Multi-Head Self-Attention │   ← every word looks at every other word (both directions)
 └───────────────────────┘
        │  + residual, LayerNorm
        ▼
 ┌───────────────────────┐
 │  Feed-Forward Network │
 └───────────────────────┘
        │  + residual, LayerNorm
        ▼
     Output (same length as input, richer vectors)
```

This block is stacked **N times** (e.g., 6 layers in the original paper, 12–24+ in modern models). Each stacked layer refines the representation further.

**Key property: bidirectional.** Every word can see every other word — both the ones before and after it. This makes the Encoder great at *understanding* full context.

**Simple example:** For the sentence "The bank raised interest rates," the Encoder lets "bank" attend to "interest rates" to correctly infer "bank" means a financial institution, not a riverbank — using context from *both sides* of the word.

---

## 5. The Decoder — Generating the Output

The Decoder's job: generate output one token at a time (e.g., generate a translation, or the next word in a story), using both what it has generated so far AND (optionally) the Encoder's understanding of the input.

### Structure of one Decoder layer:

```
Output Embeddings (shifted right) + Positional Encoding
        │
        ▼
 ┌────────────────────────────┐
 │ Masked Multi-Head Self-Attention │  ← can only look at PAST words, not future ones
 └────────────────────────────┘
        │  + residual, LayerNorm
        ▼
 ┌────────────────────────────┐
 │ Cross-Attention (Encoder-Decoder) │  ← Query from decoder, Key/Value from encoder output
 └────────────────────────────┘
        │  + residual, LayerNorm
        ▼
 ┌───────────────────────┐
 │  Feed-Forward Network │
 └───────────────────────┘
        │  + residual, LayerNorm
        ▼
   Output → passed to next layer, then to a final linear + softmax to predict next word
```

### 5.1 Masked Self-Attention

This is the same attention formula as before, but with one twist: we **mask out future positions** so the model can't "cheat" by peeking at words it hasn't generated yet.

```
Attention(Q, K, V) = softmax( (Q·Kᵀ)/√d_k + Mask ) · V
```

The mask sets scores for future positions to `-∞` before the softmax, so after softmax those probabilities become 0.

**Analogy:** imagine writing a sentence word by word with a piece of paper covering everything you haven't written yet — you can only look back at your own previous words, never ahead.

### 5.2 Cross-Attention (Encoder-Decoder Attention)

This is where the Decoder consults the Encoder's output:

```
Query  = from the Decoder (what am I trying to generate?)
Key, Value = from the Encoder output (what does the source sentence contain?)
```

**Analogy:** this is exactly like a human translator glancing back at the original sentence while writing the translation — checking "does what I'm writing so far match what's meant here?"

### 5.3 Autoregressive Generation

The Decoder produces output one token at a time:

```
Step 1: Given <START>          → predict "Le"
Step 2: Given <START> Le        → predict "chat"
Step 3: Given <START> Le chat   → predict "dort"
Step 4: Given <START> Le chat dort → predict <END>
```

Each new word is fed back in as input for the next step — hence "autoregressive."

The final prediction at each step comes from:

```
P(next word) = softmax(decoder_output · W_vocab)
```

This produces a probability distribution over the entire vocabulary; the model picks the most likely word (or samples).

---

## 6. Putting It All Together: Full Encoder-Decoder Flow

Example task: **Translate "I love cats" (English) → "J'aime les chats" (French)**

1. **Encoder:** reads "I love cats" through several stacked self-attention + FFN layers → produces a set of context vectors representing the full meaning of the sentence.
2. **Decoder (step-by-step):**
   - Starts with `<START>`.
   - Masked self-attention looks at what's been generated so far.
   - Cross-attention checks the Encoder's output ("I love cats" vectors) to decide what to say next.
   - Predicts "J'aime".
   - Feeds "J'aime" back in, predicts "les".
   - Feeds "les" back in, predicts "chats".
   - Predicts `<END>`, generation stops.

---

## 7. Three Architecture Families & When to Use Each

Modern models typically use **only one half** of the original Transformer, depending on the task.

### 7.1 Encoder-Only Models
**Examples:** BERT, RoBERTa, DistilBERT, ELECTRA

- Bidirectional — sees full context (past + future words) at once.
- Not designed to generate text; designed to **understand** it.
- Trained typically with "masked language modeling" (hide a word, predict it from context).

**Use when you need to:**
- Classify text (sentiment analysis, spam detection)
- Extract information (named entity recognition)
- Answer questions by pointing to a span in a passage (extractive QA)
- Produce embeddings for search / semantic similarity
- Any task where the input is understood as a whole and you don't need to generate new text

### 7.2 Decoder-Only Models
**Examples:** GPT-3/4, LLaMA, Claude, Mistral

- Autoregressive, causal (masked) self-attention only — no cross-attention because there's no separate encoder.
- Generates text one token at a time based only on what came before.

**Use when you need to:**
- Generate free-form text (chatbots, story writing, code generation)
- Complete/continue a prompt
- Few-shot / general reasoning tasks framed as "predict the next token"
- Most modern general-purpose LLMs use this design because it's simple, scales extremely well, and one architecture can do both understanding and generation reasonably well.

### 7.3 Encoder-Decoder (Full) Models
**Examples:** the original Transformer, T5, BART, mT5, original Google Translate models, Whisper (speech-to-text)

- Encoder builds a representation of the input; Decoder generates a different output sequence, guided by cross-attention.
- Best when input and output are **distinct sequences** that need to stay well-aligned.

**Use when you need to:**
- Machine translation (source language → target language)
- Summarization (long document → short summary)
- Speech-to-text (audio representation → text)
- Any "sequence-to-sequence" task where the output has a different structure/length/language than the input, but must remain faithful to it

### Quick comparison table

| Feature | Encoder-only | Decoder-only | Encoder-Decoder |
|---|---|---|---|
| Attention direction | Bidirectional | Causal (masked, one-way) | Bidirectional in encoder, causal in decoder |
| Best at | Understanding/classifying | Generating open-ended text | Transforming one sequence into another |
| Example models | BERT, RoBERTa | GPT, LLaMA, Claude | T5, BART, Whisper |
| Typical tasks | Classification, NER, extractive QA, embeddings | Chat, text generation, code generation | Translation, summarization, speech-to-text |
| Can generate new text? | No (not natively) | Yes | Yes |
| Sees future tokens during processing? | Yes | No | Encoder: yes / Decoder: no |

---

## 8. Simple Rule of Thumb

Ask yourself: **"What's the relationship between my input and my output?"**

- **Same text, just need to understand/label it** → Encoder-only (BERT-style)
- **Need to generate open-ended new text, possibly from a prompt** → Decoder-only (GPT-style) — this is what most modern chatbots and LLMs use
- **Need to transform one sequence into a clearly different one, and faithfulness/alignment to the source matters a lot (e.g., translation)** → Encoder-Decoder (T5/BART-style)

---

## 9. Summary Cheat Sheet

| Component | Formula | Purpose |
|---|---|---|
| Embedding | `e = Embed(token)` | Turn words into vectors |
| Positional Encoding | `PE(pos,2i)=sin(pos/10000^(2i/d))` | Inject word order |
| Self-Attention | `softmax(QKᵀ/√d_k)·V` | Let words relate to each other |
| Multi-Head | `Concat(head_1...head_h)·W_O` | Capture multiple relationship types |
| Feed-Forward | `max(0, xW1+b1)W2+b2` | Add non-linear processing per word |
| Residual + Norm | `LayerNorm(x + Sublayer(x))` | Stabilize training |
| Masked Attention | Attention + `-∞` mask on future tokens | Prevent decoder from cheating |
| Cross-Attention | Q from decoder, K/V from encoder | Let decoder consult the source |

---

## 10. Closing Notes

- All three architecture families use the **same building blocks** — the difference is *which* pieces are kept and how attention is masked.
- The dominant trend since ~2020 has been toward **decoder-only** models (GPT-family, Claude, LLaMA) because they scale well and can be adapted to almost any task via prompting — even tasks that used to require encoder-only or encoder-decoder models.
- Encoder-decoder models remain the standard choice for tasks with a very clear, structured input→output transformation, like translation or speech-to-text, where explicit cross-attention alignment is valuable.

