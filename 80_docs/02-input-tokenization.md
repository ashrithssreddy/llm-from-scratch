# Tokenization

Tokenization is the first major step in any Large Language Model (LLM) pipeline. Before a model can process text, the text must be transformed into a standardized structure the model can interpret. This process begins with **tokenization**.

---

### 🔵 What Is Tokenization?

Tokenization is the process of **splitting raw text into smaller units called tokens**. These units can be:

* Individual characters
* Words
* Subwords (most common in modern LLMs)
* Bytes

The choice depends on the tokenizer design and the model you're training.

---

### 🟦 Types of Tokenizers

#### 1. **Character-Level Tokenizers**

Breaks text into individual characters.

**Example:**

```
"hello" → ["h", "e", "l", "l", "o"]
```

**Pros:** simple, handles any text **Cons:** long sequences → slow, lacks semantic understanding

#### 2. **Word-Level Tokenizers**

Splits text into full words.

**Example:**

```
"I love pizza" → ["I", "love", "pizza"]
```

**Pros:** semantically meaningful **Cons:** huge vocabulary, struggles with unknown words

#### 3. **Subword Tokenizers (BPE, WordPiece, Unigram)**

Break words into smaller pieces based on frequency.

**Example:**

```
"playing" → ["play", "ing"]
"unbelievable" → ["un", "believ", "able"]
```

**Pros:** modern standard (GPT, LLaMA, PaLM)

* Handles rare words
* Efficient vocabulary size
* Good balance of granularity and meaning

**Cons:** more complex to train and implement

#### 4. **Byte-Level Tokenizers**

Operate directly on bytes instead of characters.

Used in GPT-2 / GPT-3 because it avoids Unicode headaches.

**Example:**

Byte-level tokenizers convert each character into its UTF‑8 byte values.

```
Input: "é"
UTF‑8 bytes: [0xC3, 0xA9]
Tokens → [195, 169]
```

Another example:

```
Input: "hello🙂"
Bytes: [104, 101, 108, 108, 111, 240, 159, 153, 130]
Tokens → same as bytes
```

**Why this helps:**

* Every possible symbol can be represented (no "unknown" tokens)
* Emojis, accents, CJK characters, and weird Unicode all map cleanly
* Vocabulary stays small (256 tokens, plus merges for BPE)

---

### 🔧 Why Tokenization Matters

Tokenization determines:

* How your model "understands" text
* Vocabulary size and memory usage
* Efficiency during training and inference
* Performance on different languages and domains

A well-designed tokenizer can significantly improve a model's capability.

---
