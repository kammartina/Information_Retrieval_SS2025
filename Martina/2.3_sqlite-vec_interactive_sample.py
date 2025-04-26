import os
import sys
import json
import apsw
import vectorlite_py
from sentence_transformers import SentenceTransformer

# ------------------------------
# Configuration
# ------------------------------
DB_PATH   = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts_multi-qa-MiniLM-L6-cos-v1.db"
MODEL_ID  = "all-MiniLM-L6-v2"
EMBED_DIM = 384
TOP_K     = 20  # number of semantic neighbors to fetch

# ------------------------------
# Semantic Query Function
# ------------------------------
def run_semantic_query(user_query: str, k: int = TOP_K, filters: dict = None):
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return []

    q_vec = model.encode([user_query])[0].tolist()
    q_json = json.dumps(q_vec)

    conn = apsw.Connection(DB_PATH)
    conn.enableloadextension(True)
    conn.loadextension(vectorlite_py.vectorlite_path())
    cur = conn.cursor()

    # Build WHERE clause
    where_clauses = []
    metadata_params = []

    if filters:
        if 'speaker' in filters:
            where_clauses.append("um.speaker = ?")
            metadata_params.append(filters['speaker'])
        if 'season' in filters:
            where_clauses.append("um.season = ?")
            metadata_params.append(filters['season'])
        if 'episode' in filters:
            where_clauses.append("um.episode = ?")
            metadata_params.append(filters['episode'])
        if 'title' in filters:
            where_clauses.append("um.title = ?")
            metadata_params.append(filters['title'])

    sql = f"""
        SELECT
          um.doc_id,
          um.season,
          um.episode,
          um.title,
          um.utterance_id,
          um.speaker,
          u.utterance_content,
          v.distance
        FROM utterance_vectors AS v
        JOIN utterance_metadata AS um ON um.utterance_id = v.rowid
        JOIN utterances         AS u  ON u.utterance_id = um.utterance_id
        WHERE knn_search(
          v.utterance_vector,
          knn_param(vector_from_json(?), ?)
        )
        {"AND " + " AND ".join(where_clauses) if where_clauses else ""}
        ORDER BY distance
        LIMIT ?
    """

    # Proper param order: [query_vector, KNN param] + metadata + limit
    params = [q_json, k] + metadata_params + [k]
    cur.execute(sql, params)
    rows = cur.fetchall()
    return rows

# ------------------------------
# Main Loop
# ------------------------------
if __name__ == "__main__":
    print("🔍 Bones Utterance Semantic Search (SQLite + Vectorlite)")
    print("Type a natural-language query or 'exit' to quit.\n")

    try:
        model = SentenceTransformer(MODEL_ID)
    except Exception as e:
        print("❌ Failed to load embedding model:", e)
        sys.exit(1)

    while True:
        try:
            query = input("🔎 Enter your search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        print("\n--- Optional Metadata Filters ---")
        speaker = input("Speaker (leave empty to skip): ").strip()
        season = input("Season (leave empty to skip): ").strip()
        episode = input("Episode (leave empty to skip): ").strip()
        title = input("Title (leave empty to skip): ").strip()

        filters = {}
        if speaker:
            filters["speaker"] = speaker
        if season:
            try:
                filters["season"] = int(season)
            except ValueError:
                print("⚠️ Invalid season number. Ignoring.")
        if episode:
            try:
                filters["episode"] = int(episode)
            except ValueError:
                print("⚠️ Invalid episode number. Ignoring.")
        if title:
            filters["title"] = title

        results = run_semantic_query(query, k=TOP_K, filters=filters)
        if not results:
            print(f"❗ No matches for '{query}' with those filters.\n")
            continue

        print(f"\n✅ Top {len(results)} match(es) for '{query}' with filters {filters}:\n")
        for rank, (doc_id, season, episode, title, utt_id, speaker, content, dist) in enumerate(results, 1):
            print(
                f"{rank}. S: {season:02d}, E: {episode:02d}, Title: '{title}', Doc ID: {doc_id}, "
                f"Utterance ID: {utt_id}, Speaker: {speaker}, dist={dist:.4f}\n"
                f"    {speaker}: {content}\n"
            )
        print("🔄 Next query or 'exit' to quit.\n")