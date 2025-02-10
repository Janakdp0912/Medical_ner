import json
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import os

# Constants
DATA_FILE = "medical_ner_data.json"
TRAIN_FILE = "train.spacy"
MODEL_PATH = "/home/atliq/Music/poc_spacy/fine_tuned_model"

def convert_to_spacy():
    """Convert JSON dataset to spaCy format and save as .spacy file."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: Dataset file '{DATA_FILE}' not found.")
        return False

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    nlp = spacy.blank("en")
    doc_bin = DocBin()

    for case in data:
        text = case["text"]
        entities = [(ent["start"], ent["end"], ent["label"]) for ent in case["entities"]]
        doc = nlp.make_doc(text)
        ents, seen_tokens = [], set()

        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span and not any(token.i in seen_tokens for token in span):
                ents.append(span)
                seen_tokens.update(token.i for token in span)

        doc.ents = ents
        doc_bin.add(doc)

    doc_bin.to_disk(TRAIN_FILE)
    print(f"✅ Dataset converted and saved as '{TRAIN_FILE}'")
    return True

def train_model():
    """Train the spaCy NER model."""
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Training file '{TRAIN_FILE}' not found. Convert dataset first.")
        return False

    try:
        nlp = spacy.blank("en")
        doc_bin = DocBin().from_disk(TRAIN_FILE)
        train_docs = list(doc_bin.get_docs(nlp.vocab))
        optimizer = nlp.begin_training()

        for i in range(10):
            losses = {}
            examples = [
                Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
                for doc in train_docs
            ]
            nlp.update(examples, drop=0.3, losses=losses)
            print(f"Iteration {i+1}, Losses: {losses}")

        nlp.to_disk(MODEL_PATH)
        print(f"✅ Fine-tuned model saved as '{MODEL_PATH}'")
        return True

    except Exception as e:
        print(f"❌ Training Error: {e}")
        return False

def predict(text):
    """Perform Named Entity Recognition (NER) on input text using the trained model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' not found. Train the model first.")
        return

    nlp = spacy.load(MODEL_PATH)
    doc = nlp(text)

    print("\nEntities from fine-tuned model:")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")

if __name__ == "__main__":
    while True:
        print("\nChoose an option:")
        print("1. Convert dataset to spaCy format")
        print("2. Train the NER model")
        print("3. Test the trained model")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            convert_to_spacy()
        elif choice == "2":
            train_model()
        elif choice == "3":
            text = input("Enter text for NER prediction: ")
            predict(text)
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")
