# Reconstructing the code used to generate the JSONL file with the exact format provided

import sqlite3
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to the SQLite database
db_path = os.getenv("SQLITE_DB_PATH")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Manually selected queries with expected keywords
queries = [
    ("Where does Dr. Brennan work?", "%Jeffersonian%"),
    ("What is a mass grave?", "%mass grave%"),
    ("What is forensic anthropology?", "%forensic anthropology%"),
    ("Who is Angela Montenegro?", "%Angela Montenegro%"),
    ("What is Booth's job?", "%FBI%"),
    ("What kind of doctor is Brennan?", "%forensic anthropologist%"),
    ("Who is Dr. Saroyan?", "%Camille Saroyan%"),
    ("What is the Jeffersonian?", "%Jeffersonian%"),
    ("What does Dr. Sweets do?", "%psychologist%"),
    ("What is a cause of death?", "%cause of death%"),
    ("What weapon was used?", "%murder weapon%"),
    ("Who found the body?", "%found the body%"),
    ("Where was the body found?", "%body was found%"),
    ("What is a greenstick fracture?", "%greenstick fracture%"),
    ("What is decomposition?", "%decomposition%"),
    ("What is the time of death?", "%time of death%"),
    ("What does Angela do?", "%facial reconstruction%"),
    ("What is a toxicologist?", "%toxicologist%"),
    ("Where did the murder happen?", "%murder took place%"),
    ("What does Hodgins study?", "%entomologist%"),
    ("Who is Pelant?", "%Pelant%"),
    ("What is blunt force trauma?", "%blunt force trauma%")
]

# Process each query to find matching utterances and format the result
evaluation_data = []

for query_text, pattern in queries:
    cursor.execute("""
        SELECT utterance_id, utterance_content FROM utterances
        WHERE utterance_content LIKE ?
        ORDER BY LENGTH(utterance_content) DESC
        LIMIT 5
    """, (pattern,))
    matches = cursor.fetchall()
    if matches:
        evaluation_data.append({
            "query": query_text,
            "gold_utterance_ids": [row[0] for row in matches],
            "expected_answer": matches[0][1]
        })

# Save to JSONL format
jsonl_path = Path("/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Evaluation_queries/bones_eval_set.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for entry in evaluation_data:
        json.dump(entry, f)
        f.write("\n")

jsonl_path.name
