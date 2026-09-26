# simple-rag

A minimal Retrieval-Augmented Generation (RAG) chatbot built with [Ollama](https://ollama.com).

Ask questions about cats and get answers grounded in the facts stored in `cat-facts.txt` — the chatbot only uses retrieved context and won't make up new information.

## How it works

1. **Load & embed** — each line of `cat-facts.txt` is treated as a chunk and embedded with `bge-base-en-v1.5` (via Ollama), stored in an in-memory vector database.
2. **Retrieve** — the user's question is embedded, cosine similarity is computed against every chunk, and the **top 3 most similar chunks** are selected.
3. **Generate** — the top 3 chunks are injected into a system prompt instructing the model to answer *only* from that context, and `Llama-3.2-1B-Instruct` streams the response token by token.

```
question → embedding → cosine similarity vs all chunks → top 3 chunks
                                                              │
                                                              ▼
                              system prompt (with context) → LLM → streamed answer
```

## Setup

1. Install [Ollama](https://ollama.com/download) and make sure it's running.
2. Pull the required models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

3. Install the Python client:

```bash
pip install ollama
```

## Run

```bash
python index.py
```

You'll see the chunks being embedded, then a prompt:

```
Loaded 118 entries
Added chunk 1/118 to the database
...
Ask me a question: How long do cats sleep?
```

The retrieved chunks (with similarity scores) are printed first, followed by the streamed chatbot response.
