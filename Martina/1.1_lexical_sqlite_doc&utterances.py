import zipfile
import sqlite3
import re
from pathlib import Path

# ------------------------------
# Configuration
# ------------------------------
ZIP_PATH     = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/5_final_cleaned_dataset.zip"
EXTRACT_DIR  = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_extracted_labeled_dataset"
DB_PATH      = "/home/mlt_ml2/Inf_Retrieval_Project_Martina_Sandra/Martina/1_bones_scripts_multi-qa-MiniLM-L6-cos-v1.db"

# ------------------------------
# 1. Unzip dataset
# ------------------------------
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    zf.extractall(EXTRACT_DIR)
print(f"Dataset extracted to: {EXTRACT_DIR}")

# ------------------------------
# 2. Connect & create tables
# ------------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# DOCS (episodes) table
cursor.execute('''
CREATE TABLE IF NOT EXISTS docs (
    doc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    season          INTEGER,
    episode         INTEGER,
    title           TEXT,
    doc_content     TEXT,
    UNIQUE(season, episode, title)
);
''')

# UTTERANCES table (all rows - "speaker: text")
cursor.execute('''
CREATE TABLE IF NOT EXISTS utterances (
    utterance_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    season              INTEGER,
    episode             INTEGER,
    title               TEXT,
    speaker             TEXT,
    utterance_content   TEXT
);
''')

conn.commit()

# ------------------------------
# 3. Parser helpers
# ------------------------------

def parse_episode_header(lines):
    title = lines[0].strip()
    m = re.match(r"season\s*(\d+),\s*episode\s*(\d+)", lines[1].lower())
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
    else:
        season = episode = None
    return title, season, episode

utterance_re = re.compile(r"^([^:\[\]]+?):\s*(.+)")

# ------------------------------
# 4. Populate data
# ------------------------------

for txt_path in Path(EXTRACT_DIR).glob("*.txt"):
    lines = txt_path.read_text(encoding='utf-8').splitlines()

    # Insert episode metadata
    title, season, episode = parse_episode_header(lines)
    script = "\n".join(lines[2:]).strip() # script is everything after the header

    # Insert into docs
    cursor.execute(
        "INSERT OR IGNORE INTO docs (season, episode, title, doc_content) VALUES (?, ?, ?, ?)",
        (season, episode, title, script)
    )
    conn.commit()

    # Parse utterances (skip first two lines)
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        m = utterance_re.match(line)
        if not m:
            continue  # skip non-utterance lines
        speaker = m.group(1).strip()
        content = m.group(2).strip()

        # Insert utterance
        cursor.execute(
            "INSERT INTO utterances (season, episode, title, speaker, utterance_content) VALUES (?, ?, ?, ?, ?)",
            (season, episode, title, speaker, content)
        )

# Commit utterances
conn.commit()

print("Docs and utterance level indexes built successfully.")