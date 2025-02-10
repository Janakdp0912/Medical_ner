import spacy
import streamlit as st
import os

# Constants
MODEL_PATH = "/home/atliq/Music/poc_spacy/fine_tuned_model"

def predict(text):
    """Perform Named Entity Recognition (NER) on input text using the trained model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model '{MODEL_PATH}' not found. Train the model first.")
        return

    nlp = spacy.load(MODEL_PATH)
    doc = nlp(text)

    st.write("Entities from fine-tuned model:")
    for ent in doc.ents:
        st.write(f"{ent.text} ({ent.label_})")

def main():
    st.title("Medical NER Prediction with finetuned spacy model")

    # Input text for prediction
    input_text = st.text_area("Enter text for NER prediction:")

    if st.button("Run Prediction"):
        if input_text:
            predict(input_text)
        else:
            st.error("Please enter some text for prediction.")

if __name__ == "__main__":
    main()
