import uuid

import chromadb
from sentence_transformers import SentenceTransformer

# Load the model
qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# The queries and documents to embed
queries = ["What is the capital of China?", "What is the force that large bodies of mass exert?"]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = qwen_model.encode(queries, prompt_name="query").tolist()
document_embeddings = qwen_model.encode(documents).tolist()

ids = [str(uuid.uuid4()) for _ in documents]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="my_collection", metadata={"hnsw:space": "cosine"}
)


def add_documents(documents, document_embeddings, ids, collection, top_k):

    collection.add(documents=documents, embeddings=document_embeddings, ids=ids)

    results = collection.query(query_embeddings=query_embeddings, n_results=top_k)

    return results


results = add_documents(documents, document_embeddings, ids, collection, 2)

for query, docs, distances, ids in zip(
    queries, results["documents"], results["distances"], results["ids"]
):
    print(f"\nQuery: {query}")

    for doc, distance, doc_id in zip(docs, distances, ids):
        print(f"ID: {doc_id}")
        print(f"Document: {doc}")
        print(f"Distance: {distance:.4f}")
