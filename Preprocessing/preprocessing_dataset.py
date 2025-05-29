import os
import re
import random
from pathlib import Path
from canonical_speaker_names import character_name_variants
from censored_words import censored_words
from dotenv import load_dotenv
from tqdm import tqdm


class TranscriptCleaner:
    def __init__(self, input_folder, output_folder):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.remaining_asterisk_words = set()

    def replace_censored_words(self, text):
        for censored, replacement in tqdm(sorted(censored_words.items(), key=lambda x: -len(x[0])), desc="Replacing censored words"):
            pattern = re.compile(re.escape(censored), re.IGNORECASE)
            text = pattern.sub(replacement, text)
        return text

    def find_censored_words(self, text):
        return set(re.findall(r'(?<!\w)\D\w*?\+\w?\D(?!\w)', text))

    def normalize_character_names(self, text):
        def normalize_line(line):
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                speaker, content = parts
                normalized = character_name_variants.get(speaker.strip().lower(), speaker.strip().lower())
                return f"{normalized}: {content.strip()}"
            return line

        lines = text.splitlines()
        return "\n".join(normalize_line(line) for line in tqdm(lines, desc="Normalizing names"))

    def process_transcript(self, file_name, text):
        lines = text.splitlines()
        content = lines[2:] if len(lines) >= 2 else lines

        for i, line in enumerate(content):
            if any(keyword in line.lower() for keyword in ["written by", "directed by", "transcribed by", "disclaimer",
                                                            "teaser", "produced by", "screenplay", "airdate"]):
                continue
            if '♪' in line or 'adsbygoogle' in line or 'googlesyndication' in line:
                continue
            content = content[i:]
            break

        base_file = os.path.basename(file_name)
        if base_file.startswith("._") or "__MACOSX" in file_name:
            return None

        match = re.match(r"(\d{2})x(\d{2})-(.+)\.txt", base_file)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))
            title = match.group(3).replace("-", " ").lower()
            header = [title, f"season {season}, episode {episode}"]
        else:
            header = ["Unknown Episode Title", "Unknown Season and Episode"]

        return "\n".join(header + content).strip()

    def clean_dataset(self):
        files = list(self.input_folder.glob("*.txt"))
        for file_path in tqdm(files, desc="Cleaning files"):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

            text = self.replace_censored_words(text)
            text = text.lower()
            self.remaining_asterisk_words |= self.find_censored_words(text)
            text = self.normalize_character_names(text)
            text = self.process_transcript(file_path.name, text)

            if text:
                text = "\n".join(line for line in text.splitlines() if line.strip())
                output_path = self.output_folder / file_path.name
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(text)

        if self.remaining_asterisk_words:
            print("\nRemaining censored words:")
            for word in sorted(self.remaining_asterisk_words):
                print(word)
        else:
            print("\nAll censored words were replaced successfully.")

        print(f"\nCleaned dataset saved to: {self.output_folder}")

    def preview_random_sample(self):
        txt_files = list(self.output_folder.glob("*.txt"))
        if not txt_files:
            print("No cleaned files to preview.")
            return

        sample_file = random.choice(txt_files)
        print(f"\nSample from: {sample_file.name}\n{'='*50}")
        with open(sample_file, 'r', encoding='utf-8') as f:
            for line in f.readlines()[:50]:
                print(line.strip())


def main():
    load_dotenv()
    input_folder_path = os.getenv("INPUT_DATASET")
    output_folder_path = os.getenv("OUTPUT_DATASET")

    print("INPUT_DATASET =", input_folder_path)
    print("OUTPUT_DATASET =", output_folder_path)
    if not input_folder_path or not output_folder_path:
        raise ValueError("Missing INPUT_DATASET or OUTPUT_DATASET environment variables.")

    cleaner = TranscriptCleaner(input_folder_path, output_folder_path)
    cleaner.clean_dataset()
    cleaner.preview_random_sample()


if __name__ == '__main__':
    main()