import chromadb

# --------------------
# Configuration
# --------------------
CHROMA_DIR = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/3_chroma_db"
COLLECTION_NAME = "utterance_vectors_chroma"

# --------------------
# Load Chroma Collection
# --------------------
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# --------------------
# User input
# --------------------
user_query = input("\nEnter your search query: ").strip()

if not user_query:
    print("❌ No input provided. Exiting.")
    exit()

print("\n--- Optional Filters ---")
speaker = input("Speaker (leave empty to skip): ").strip()
season = input("Season (leave empty to skip): ").strip()
episode = input("Episode (leave empty to skip): ").strip()
title = input("Title (leave empty to skip): ").strip()

# Build the filter dictionary
filter_metadata = {}
if speaker:
    filter_metadata["speaker"] = speaker
if season:
    filter_metadata["season"] = int(season)  # Season must be int
if episode:
    filter_metadata["episode"] = int(episode)  # Episode must be int
if title:
    filter_metadata["title"] = title

n_results = 10  # Number of similar utterances to retrieve

print(f"\n🔎 Searching for: '{user_query}' with filters {filter_metadata}...\n")

# --------------------
# Query
# --------------------
results = collection.query(
    query_texts=[user_query],
    n_results=n_results,
    where=filter_metadata if filter_metadata else None
)

# --------------------
# Display Results
# --------------------
if not results['documents'][0]:
    print("No results found.")
else:
    for i in range(len(results['documents'][0])):
        print(f"Result {i+1}:")
        print("  Utterance:", results['documents'][0][i])
        print("  Metadata :", results['metadatas'][0][i])
        print("  Distance :", results['distances'][0][i])
        print()