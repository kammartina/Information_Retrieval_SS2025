"""Add commentMore actions
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
from dotenv import load_dotenv
from tqdm import tqdm

class ChromaUtteranceIndexer:
    def __init__(self):
        load_dotenv()
        self.db_path = os.getenv("SQLITE_DB_PATH")
        self.chroma_dir = os.getenv("CHROMA_DB_PATH")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.batch_size = int(os.getenv("BATCH_SIZE"))

        self.conn = self.connect_sqlite()
        self.client = self.init_chroma_client()
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.prepare_collection()

    def connect_sqlite(self):
        print("Connecting to SQLite database...")
        return apsw.Connection(self.db_path)
    
    def init_chroma_client(self):
        print("Initializing Chroma client collection...")
        print("Using ChromaDB persist directory:", os.path.abspath(self.chroma_dir))
        return chromadb.PersistentClient(path=self.chroma_dir)
    
    def prepare_collection(self):
        existing_collections = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing_collections:
            print(f"Deleting existing collection: '{self.collection_name}'")
            self.client.delete_collection(name=self.collection_name)
        return self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
    
    def load_utterances_with_metadata(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT 
                u.utterance_id, 
                u.utterance_content, 
                d.season, 
                d.episode, 
                d.title, 
                u.speaker
            FROM utterances u
            JOIN docs d ON u.doc_id = d.doc_id
            WHERE u.utterance_content IS NOT NULL 
            AND TRIM(u.utterance_content) != '';
        """)
        for utt_id, text, season, episode, title, speaker in cur:
            metadata = {
                "season": season,
                "episode": episode,
                "title": title,
                "speaker": speaker
            }
            yield str(utt_id), text.strip(), metadata

    def embed_and_index(self):
        print("Embedding and indexing utterances with metadata...")
        
        ids, texts, metadatas = [], [], []
        total_inserted = 0

        utterances = list(self.load_utterances_with_metadata())
        for i, (utt_id, text, metadata) in enumerate(tqdm(utterances, desc="Indexing utterances"), 1):
            ids.append(utt_id)
            texts.append(text)
            metadatas.append(metadata)

            if len(ids) >= self.batch_size:
                self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
                total_inserted += len(ids)
                print(f"  • {total_inserted} utterances embedded...")
                ids.clear()
                texts.clear()
                metadatas.clear()

        # Insert remaining
        if ids:
            self.collection.add(documents=texts, ids=ids, metadatas=metadatas)
            total_inserted += len(ids)
            print(f"  • {total_inserted} utterances embedded (final batch)...")

        print(f"Finished creating Chroma collection.")
        print(f"Total embedded utterances: {self.collection.count()}")
    
    def run(self):
        self.embed_and_index()
        print("Chroma Utterance Indexing completed successfully.")
        self.conn.close()

def main():
    indexer = ChromaUtteranceIndexer()
    indexer.run()

if __name__ == "__main__":
    main()