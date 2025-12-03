# Encoding (stoi/itos):

Encoding is the step that follows tokenization in the LLM pipeline. After text is split into tokens, those tokens need to be converted into a numerical format so the model can understand and process them. This transformation is called **encoding**.

---

### 🔵 What Is Encoding?

Encoding is the process of **mapping each token to an integer ID**. Neural networks cannot operate on text directly—they only work with numbers—so encoding is essential.

In practice, encoding uses two key mappings:

* **stoi** → string‑to‑index
* **itos** → index‑to‑string (reverse)

These mappings together form the vocabulary of your tokenizer.

---

### 🟩 stoi: String → Index

This mapping assigns each token a unique integer.

**Example:** Suppose your tokenizer produced these tokens:

```
["h", "e", "l", "l", "o"]
```

You might create this mapping:

```
stoi = {
  "h": 0,
  "e": 1,
  "l": 2,
  "o": 3
}
```

Now encoding becomes:

```
"hello" → [0, 1, 2, 2, 3]
```

---

### 🟧 itos: Index → String

This is the reverse mapping.

```
itos = {
  0: "h",
  1: "e",
  2: "l",
  3: "o"
}
```

Decoding works like this:

```
[0, 1, 2, 2, 3] → "hello"
```

---

### 🛠 A Full Practical Example

Imagine the input text is:

```
"I love pizza!"
```

##### Step 1: Tokenization

Word‑level example:

```
["I", "love", "pizza", "!"]
```

##### Step 2: Build stoi

```
stoi = {
  "I": 0,
  "love": 1,
  "pizza": 2,
  "!": 3
}
```

##### Step 3: Encode

```
[0, 1, 2, 3]
```

##### Step 4: Decode (using itos)

```
"I love pizza!"
```

---

### 🔢 Encoding in Real LLMs

Modern LLMs do not manually craft these dictionaries. Instead:

* GPT models use **BPE tokenizers** with prebuilt vocabularies
* Each subword piece has a stable, published integer ID
Example (GPT‑2):

```
"playing" → ["play", "ing"] → [1327, 352]
```

Below are the standard publicly available vocabularies and merge rules used by well‑known models:

* **GPT‑2 Vocabulary (vocab.json)** — [https://huggingface.co/openai-community/gpt2/blob/main/vocab.json](https://huggingface.co/openai-community/gpt2/blob/main/vocab.json)
* **GPT‑2 BPE Merge Rules (merges.txt)** — [https://huggingface.co/openai-community/gpt2/blob/main/merges.txt](https://huggingface.co/openai-community/gpt2/blob/main/merges.txt)
* **GPT‑Neo Vocabulary** — [https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/main/vocab.json](https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/main/vocab.json)
* **GPT‑Neo Merge Rules** — [https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/main/merges.txt](https://huggingface.co/EleutherAI/gpt-neo-1.3B/blob/main/merges.txt)
* **LLaMA SentencePiece Model (tokenizer.model)** — [https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/tokenizer.model](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/tokenizer.model)
* **LLaMA Vocabulary (tokenizer.json)** — [https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/tokenizer.json](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/tokenizer.json)

These are the exact files used during training of those models. All IDs and tokens match the model weights.

---

### 🟦 Why Encoding Matters

Encoding:

* Translates human‑readable text into model‑readable numbers
* Ensures consistency in how tokens are represented
* Allows fast lookup during training
* Defines the size of the embedding layer and logits output

---
