import json
import apsw
import vectorlite_py

DB_PATH = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts_multi-qa-MiniLM-L6-cos-v1.db"
EMBED_DIM = 384
UTT_COUNT = 61507  # total utterances in your DB

def main():
    conn = apsw.Connection(DB_PATH)
    conn.enableloadextension(True)
    conn.loadextension(vectorlite_py.vectorlite_path())
    cur = conn.cursor()

    # Build a zero-vector to probe the index
    zero = json.dumps([0.0] * EMBED_DIM)

    # 1) Count how many entries your HNSW index actually has
    cur.execute("""
      SELECT COUNT(*) 
      FROM utterance_vectors
      WHERE knn_search(
        utterance_vector,
        knn_param(vector_from_json(?), ?)
      );
    """, (zero, UTT_COUNT))
    count = cur.fetchone()[0]
    print(f"Vectors indexed: {count} / {UTT_COUNT}")

    if count == 0:
        print("⚠️  No vectors found—re-run your build_with_apsw script before querying.")
        return

    # 2) Pull out one sample (closest to the zero vector)
    cur.execute("""
      SELECT rowid, distance 
      FROM utterance_vectors
      WHERE knn_search(
        utterance_vector,
        knn_param(vector_from_json(?), 1)
      );
    """, (zero,))
    rowid, dist = cur.fetchone()
    print(f"\nSample rowid={rowid}, distance={dist:.4f}")

    # 3) Fetch its text
    cur.execute("SELECT utterance_content FROM utterances WHERE utterance_id = ?;", (rowid,))
    text = cur.fetchone()[0]
    print("Content:", text)

    conn.close()

if __name__ == "__main__":
    main()
