import ollama


# --- Load the dataset ---

# Each line of the file is one "chunk" of knowledge (a single cat fact).
dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')



# --- Implement the retrieval system ---

# Two local Ollama models:
# - an embedding model to vectorize chunks and queries
# - a chat model to generate the final answer
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
# The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
# (In-memory "vector database" — fine for a demo, swap for a real vector store in production.)
VECTOR_DB = []

def add_chunk_to_database(chunk):
  # Embed the chunk and store it alongside its vector
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

# Vectorize the whole dataset at startup
for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  # dot(a, b) / (|a| * |b|) — 1.0 means identical direction (very similar), 0.0 means unrelated
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  # Embed the query with the SAME model used for the chunks,
  # otherwise the vectors live in different spaces and similarity is meaningless
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]



# --- Chatbot ---

input_query = input('Ask me a question: ')
# Find the 3 chunks most relevant to the question
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

# RAG step: inject the retrieved chunks into the system prompt.
# The model is told to answer ONLY from this context, so it can't hallucinate new facts.
# (retrieved_knowledge is consumed here — the chunks end up inside instruction_prompt.)
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''
# print(instruction_prompt)

# Ask the chat model. stream=True returns a generator that yields the response
# piece by piece instead of waiting for the full answer.
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},  # contains the top 3 chunks
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

# print the response from the chatbot in real-time
# end='' keeps everything on one line, flush=True shows each token the moment it arrives
print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
