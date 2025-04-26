"""
CREATING A PERSISTENT CHROMADB COLLECTION

This script creates a persistent ChromaDB collection from utterances stored in an SQLite database and their metadata.

- collection name "utterance_vectors_chroma" permanently stored in the CHROMA_DIR
- embedding function: DefaultEmbeddingFunction
    - by default, it uses the Hugging Face model "sentence-transformers/all-MiniLM-L6-v2"
- batch size: 500 utterances
- metric: generally uses Euclidean distance or cosine similarity
    - here by default Euclidean distance (L2)
    -  perfect match: 0
    - the lower the distance, the more similar the vectors
"""

"""
METRIC DISTANCE
- Euclidean distance (L2) looks at "stright-line distance" between two points in 384-dimensional space
- Cosine similarity looks at "angle between two vectors", ignoring magnitude


--How to find out which metric is used?
collection = client.get_collection(name="utterance_vectors_chroma")
print(collection.metadata)


--How to specify the metric?
collection = client.create_collection(
    name="utterance_vectors_chroma",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # <-- Here you set distance metric
)

- Euclidean distance --> "hnsw:space": "l2" (default)
- Cosine Similarity --> "hnsw:space": "cosine"
- Inner Product --> "hnsw:space": "ip"
...
"""

import os
import apsw
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------
# Configuration
# ------------------------------
DB_PATH = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts.db"
COLLECTION_NAME = "utterance_vectors_chroma"
CHROMA_DIR = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/3_chroma_db"
BATCH_SIZE = 500


# Helper function to load utterances and metadata from SQLite database
def load_utterances_with_metadata(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT utterance_id, utterance_content, season, episode, title, speaker
        FROM utterances
        WHERE utterance_content IS NOT NULL AND TRIM(utterance_content) != '';
    """)
    for utt_id, text, season, episode, title, speaker in cur:
        metadata = {
            "season": season,
            "episode": episode,
            "title": title,
            "speaker": speaker
        }
        yield str(utt_id), text.strip(), metadata


def main():
    # we need to connect to the SQLite database to read utterances
    # and to ChromaDB to create a collection and add vectors
    print("Connecting to SQLite database...")
    conn = apsw.Connection(DB_PATH)

    print("Using the default embedding function...")
    ef = embedding_functions.DefaultEmbeddingFunction()

    # persistent vector database - saving and loading to/from disk
    print("Initializing Chroma client and collection...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    print("Using ChromaDB persist directory:", os.path.abspath(CHROMA_DIR))

    # Remove old collection if exists
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Deleting existing collection: '{COLLECTION_NAME}'")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    print(f"Created new collection: '{COLLECTION_NAME}'\n")

    print("Embedding and indexing utterances with metadata...")
    
    ids, texts, metadatas = [], [], []
    total_inserted = 0

    for i, (utt_id, text, metadata) in enumerate(load_utterances_with_metadata(conn), 1):
        ids.append(utt_id)
        texts.append(text)
        metadatas.append(metadata)

        if len(ids) >= BATCH_SIZE:
            collection.add(documents=texts, ids=ids, metadatas=metadatas)
            total_inserted += len(ids)
            print(f"  • {total_inserted} utterances embedded...")
            ids.clear()
            texts.clear()
            metadatas.clear()

    # Insert remaining
    if ids:
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
        total_inserted += len(ids)
        print(f"  • {total_inserted} utterances embedded (final batch)...")

    print(f"Finished creating Chroma collection.")
    print(f"Total embedded utterances: {collection.count()}")

if __name__ == "__main__":
    main()