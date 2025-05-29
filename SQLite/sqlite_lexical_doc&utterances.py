import sqlite3
import re
from dotenv import load_dotenv
import os
from pathlib import Path

#------------------
class DatasetLoader:
    def __init__(self, cleaned_dataset_path, sqlite_db_path):
        self.cleaned_dataset_path = Path(cleaned_dataset_path)
        self.sqlite_db_path = sqlite_db_path
        self.conn = sqlite3.connect(self.sqlite_db_path)
        self.cursor = self.conn.cursor()
        self.utterance_re = re.compile(r"^([^:\[\]]+?):\s*(.+)")
    
    def setup_tables(self):
        # Create docs table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS docs (
                doc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                season          INTEGER,
                episode         INTEGER,
                title           TEXT,
                doc_content     TEXT,
                UNIQUE(season, episode, title)
            );
        ''')

        # Create utterances table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS utterances (
                utterance_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id              INTEGER,
                speaker             TEXT,
                utterance_content   TEXT,
                FOREIGN KEY(doc_id) REFERENCES docs(doc_id)
            );
        ''')
        self.conn.commit()
    
    def parse_episode_header(self, lines):
        title = lines[0].strip()
        match = re.match(r"season\s*(\d+),\s*episode\s*(\d+)", lines[1].lower())
        if match:
            return title, int(match.group(1)), int(match.group(2))
        return title, None, None
    
    def insert_doc(self, season, episode, title, doc_content):
        self.cursor.execute(
            "INSERT OR IGNORE INTO docs (season, episode, title, doc_content) VALUES (?, ?, ?, ?)",
            (season, episode, title, doc_content)
        )
        self.conn.commit()
        self.cursor.execute(
            "SELECT doc_id FROM docs WHERE season = ? AND episode = ? AND title = ?", 
            (season, episode, title)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def insert_utterance(self, doc_id, speaker, utterance_content):
        self.cursor.execute(
            "INSERT INTO utterances (doc_id, speaker, utterance_content) VALUES (?, ?, ?)",
            (doc_id, speaker, utterance_content)
        )
    
    def load_from_folder(self):
        txt_files = list(self.cleaned_dataset_path.glob("*.txt"))
        for file_path in txt_files:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
                self.process_file(lines)
        
        self.conn.commit()
        self.conn.close()
        print("Finished loading data from folder.")

    def process_file(self, lines):
        if len(lines) < 3:
            return
        
        title, season, episode = self.parse_episode_header(lines)
        doc_content = "\n".join(lines[2:]).strip()  # everything after the header
        doc_id = self.insert_doc(season, episode, title, doc_content)

        for line in lines[2:]:
            line = line.strip()
            if not line:
                continue
            match = self.utterance_re.match(line)
            if not match:
                continue
            speaker= match.group(1).strip()
            utterance_content = match.group(2).strip()
            self.insert_utterance(doc_id, speaker, utterance_content)

def main():
    # Load environment variables
    load_dotenv()
    cleaned_dataset_path = os.getenv("OUTPUT_DATASET")
    sqlite_db_path = os.getenv("SQLITE_DB_PATH")

    if not cleaned_dataset_path or not sqlite_db_path:
        raise ValueError("OUTPUT_DATASET and SQLITE_DB_PATH must be set in the environment variables.")
    
    # Initialize and setup dataset loader
    loader = DatasetLoader(cleaned_dataset_path, sqlite_db_path)
    loader.setup_tables()
    loader.load_from_folder()
        
if __name__ == "__main__":
    main()