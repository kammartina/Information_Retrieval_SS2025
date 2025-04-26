"""
CREATING SQLite-vec EXTENSION

This script uses APSW instead of sqlite3 to create a SQLite extension that allows for vector operations.
(because sqlite3 is old, it does not support loading extensions)
"""

"""
--> SQLite (with Vectorlite) can store and search embeddings, but it cannot create them. Therefore 
you need to use an embedding model to create the embeddings first.

Example:
An “embedding API” (like OpenAI’s openai.Embedding.create) is simply a service you send raw text to, and it returns back 
a fixed-length list of numbers (a vector) that encodes the meaning of that text in a way a computer can work with numerically.
-> openai is NOT free of charge

I used:
SentenceTransformer is from the sentence-transformers package from Hugging Facw; it gives 
pretrained models that convert text into vectors. I used: "sentence-transformers/all-MiniLM-L6-v2" model.

-----------

Vectorlite:
--> is a lightweight vector search extension for SQLite — it lets you:
- Define vector columns (embedding float32[384])
- Insert vectors
- Search using approximate nearest neighbor (ANN) via HNSW
- Do KNN queries like knn_search(...)
!! But it does not know how to create vectors, so you need to use an embedding model to create the vectors first.
    - a.k.a.: hot to turn "blunt force trauma" into [-0.023, 0.512, ..., 0.148]  # A 384-dimension float vector
"""

"""
What kind of queries can I run?

- single keywords - skull, grave, airport
- short phrases - mass grave, blunt force trauma (must be in double quotes to keep order)
- mini-sentences or questions - What is a skull?, diving in a mass grave
- synonyms or paraphrases - diggin up the bodies
"""

"""
How is the similarity measured?
- sentence-transformers/all-MiniLM-L6-v2 --> cosine distance
- sentence-transformers/multi-qa-MiniLM-L6-cos-v1 --> cosine distance
--> the vectors from sentence-transformers are normalized to unit vectors, 
    and Vectorlite uses COSINE DISTANCE (cos.dist = 1 - cos.similarity), so the range is 0 to 2.

COSINE SIMILARITY
- 1 - exact match
- 0 - orthogonal (no similarity)
- -1 - opposite (completely dissimilar)

COSINE DISTANCE
- 0 - exact match
- 1 - orthogonal (no similarity)
- 2 - opposite (completely dissimilar)
"""

import json
import apsw
import vectorlite_py
from sentence_transformers import SentenceTransformer

DB_PATH     = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts_multi-qa-MiniLM-L6-cos-v1.db"
EMBED_DIM   = 384
UTT_COUNT   = 61507             # total utterances
HNSW_PARAMS = f"hnsw(max_elements={UTT_COUNT})"

def main():
    print("Loading embedding model…")
    model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    print("Opening APSW connection & loading vectorlite…")
    conn = apsw.Connection(DB_PATH)
    conn.enableloadextension(True)
    conn.loadextension(vectorlite_py.vectorlite_path())

    # Two cursors so SELECT and INSERT don’t clobber each other
    read_cur  = conn.cursor()
    write_cur = conn.cursor()

    print("(Re)creating `utterance_vectors` VTABLE…")
    write_cur.execute("DROP TABLE IF EXISTS utterance_vectors;")
    write_cur.execute(f"""
      CREATE VIRTUAL TABLE utterance_vectors
        USING vectorlite(
          utterance_vector float32[{EMBED_DIM}],
          hnsw(max_elements=61507),
          'utterance_vectors.hnsw'
        );
    """)

    print("(Re)creating `utterance_metadata` table…")
    write_cur.execute("DROP TABLE IF EXISTS utterance_metadata;")
    write_cur.execute("""
        CREATE TABLE utterance_metadata (
            utterance_id    INTEGER PRIMARY KEY,
            doc_id          INTEGER,
            season          INTEGER,
            episode         INTEGER,
            title           TEXT,
            speaker         TEXT,
            FOREIGN KEY(utterance_id) REFERENCES utterances(utterance_id),
            FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
        );
    """)

    # !!! Start an explicit transaction
    write_cur.execute("BEGIN;")

    print(f"Indexing {UTT_COUNT} utterances…")
    # Join utterances with docs to get doc_id
    read_cur.execute("""
        SELECT 
            u.utterance_id,
            d.doc_id,
            u.season,
            u.episode,
            u.title,
            u.speaker,
            u.utterance_content
        FROM utterances u
        JOIN docs d ON u.season = d.season AND u.episode = d.episode AND u.title = d.title;
    """)

    n = 0
    for row in read_cur:
        utt_id, doc_id, season, episode, title, speaker, content = row
        content = (content or "").strip()
        if not content:
            continue

        vec = model.encode([content])[0].tolist()

        # Insert into vector table
        write_cur.execute(
            "INSERT OR REPLACE INTO utterance_vectors(rowid, utterance_vector) VALUES (?, vector_from_json(?));",
            (utt_id, json.dumps(vec))
        )

        # Insert into metadata table
        write_cur.execute(
            "INSERT INTO utterance_metadata (utterance_id, doc_id, season, episode, title, speaker) VALUES (?, ?, ?, ?, ?, ?);",
            (utt_id, doc_id, season, episode, title, speaker)
        )

        n += 1
        if n % 5000 == 0:
            print(f"  • {n} done…")
    
    # !!! Commit the transaction
    write_cur.execute("COMMIT;")

    print(f"Finished indexing {n} vectors.")

    print("Creating utterance embeddings was successful.")

    conn.close()

if __name__ == "__main__":
    main()