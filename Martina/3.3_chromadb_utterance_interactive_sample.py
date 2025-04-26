import os
import sys
import chromadb
from chromadb.config import Settings

# ------------------------------
# Configuration
# ------------------------------
CHROMA_DIR = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/3_chroma_db"
COLLECTION_NAME = "utterance_vectors_chroma"
TOP_K = 20  # number of semantic neighbors to fetch

# ------------------------------
# Helper function
# ------------------------------
def run_chroma_query(user_query: str, k: int = TOP_K, filters: dict = None):
    """Run a semantic query over ChromaDB collection."""
    if not user_query.strip():
        return []

    results = collection.query(
        query_texts=[user_query],
        n_results=k,
        where=filters if filters else None
    )
    
    # Flatten results into list of dicts
    matches = []
    for i in range(len(results['documents'][0])):
        match = {
            'utterance': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        }
        matches.append(match)
    return matches

# ------------------------------
# Main interactive loop
# ------------------------------
if __name__ == "__main__":
    print("🔍 Bones Utterance Semantic Search (ChromaDB Version)")
    print("Type a natural-language query (e.g. “mass grave”) or 'exit' to quit.\n")

    # Connect to ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB or collection: {e}", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            user_query = input("🔎 Enter your search query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        print("\n--- Optional Metadata Filters ---")
        speaker = input("Speaker (leave empty to skip): ").strip()
        season = input("Season (leave empty to skip): ").strip()
        episode = input("Episode (leave empty to skip): ").strip()
        title = input("Title (leave empty to skip): ").strip()

        # Build filter dictionary
        filter_metadata = {}
        if speaker:
            filter_metadata["speaker"] = speaker
        if season:
            try:
                filter_metadata["season"] = int(season)
            except ValueError:
                print("⚠️ Invalid season number. Ignoring.")
        if episode:
            try:
                filter_metadata["episode"] = int(episode)
            except ValueError:
                print("⚠️ Invalid episode number. Ignoring.")
        if title:
            filter_metadata["title"] = title

        results = run_chroma_query(user_query, k=TOP_K, filters=filter_metadata)
        if not results:
            print(f"❗ No semantic matches for '{user_query}' with those filters.\n")
            continue

        print(f"\n✅ Top {len(results)} semantic match(es) for '{user_query}' with filters {filter_metadata}:\n")
        for rank, match in enumerate(results, 1):
            meta = match['metadata']
            season = meta.get('season', 0)
            episode = meta.get('episode', 0)
            title = meta.get('title', 'Unknown Title')
            speaker = meta.get('speaker', 'Unknown Speaker')
            utt_id = meta.get('utterance_id', 'N/A')  # Optional if stored
            distance = match['distance']
            content = match['utterance']

            print(
                f"{rank}. S: {season:02d}, E: {episode:02d}, Title: '{title}'"
                f", Speaker: {speaker}, dist={distance:.4f}\n"
                f"    {speaker}: {content}\n"
            )
        print("🔄 Next query or 'exit' to quit.\n")