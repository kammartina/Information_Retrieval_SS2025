import sqlite3
import os

DB_PATH = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts_multi-qa-MiniLM-L6-cos-v1.db"

def run_query(user_query):
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return []
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        sql = """
        SELECT 
            d.doc_id,
            u.season,
            u.episode,
            u.title,
            u.utterance_id,
            u.speaker,
            u.utterance_content
        FROM utterances u
        JOIN docs d ON u.season = d.season AND u.episode = d.episode AND u.title = d.title
        WHERE u.utterance_content LIKE ?
        ORDER BY u.season, u.episode, u.utterance_id;
        """

        cur.execute(sql, (f"%{user_query}%",))
        rows = cur.fetchall()
        conn.close()
        return rows
    
    except sqlite3.OperationalError as e:
        print(f"❌ SQLite error: {e}")
        print("💡 Valid examples: skull, skull AND trauma, \"blunt force trauma\", skull NEAR trauma")
        return []

if __name__ == "__main__":
    print("🔍 Bones Utterance Search (FTS5)")
    print("Type a query (e.g. skull, \"blunt force trauma\", NEAR, AND/OR/NOT). Use quotes to match the exact phrase. Type 'exit' to quit.\n")
    
    while True:
        try:
            q = input("🔎 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        results = run_query(q)
        if not results:
            print(f"❗ No matches for '{q}'\n")
            continue

        print(f"\n✅ {len(results)} match(es) for '{q}':\n")
        for doc_id, season, episode, title, utterance_id, speaker, content in results:
            print(
                f"➤  S: {season:02d}, E: {episode:02d}, Title: '{title}', Doc ID: {doc_id}, Utterance ID: {utterance_id}\n"
                f"    {speaker}: {content}\n"
            )
        print("🔄 Next query or 'exit' to quit.\n")

#     """
# case-sentsitive by default with FTS --> Skull === skull

# phrases must be in double quotes --> "skull trauma"

# combine as many operators as you like --> 
    # "mass grave" AND (skull OR bone)  
    # skull* AND NOT (hemmorrhage OR fracture)

# If you type plain English (e.g. “find all utterances with the word skull”), FTS will treat each word (find, all, utterances, with, the, word, skull) 
# as separate tokens and try to match all of them (so you’ll likely get zero hits)
