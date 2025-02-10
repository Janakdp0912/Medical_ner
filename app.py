# Import necessary libraries
import spacy
import os
import json
import streamlit as st
from groq import Groq
import pandas as pd

# Constants
MODEL_PATH = "fine_tuned_model"
API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Function to extract entities using LLM
def extract_entities_llm(text):
    system_prompt = """
    Extract medical conditions, medicine, and pathogens.
    """
    user_prompt = f"""
    Text: {text}
    Return entities in JSON format with keys: "entity", "type", and "position" (start index). Example:
    {{
     "entities": [
        {{"entity": "Cancer", "type": "MEDICAL CONDITION", "position": 0}},
        {{"entity": "Diabetes", "type": "MEDICAL CONDITION", "position": 30}}
      ]
    }}
    """

    # Call Groq API
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5,
        response_format={"type": "json_object"}
    )

    # Parse and return the response
    response = completion.choices[0].message.content
    return json.loads(response)

# Function to extract entities using pre-trained SpaCy model
def extract_entities_spacy_pretrained(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [{"entity": ent.text, "type": ent.label_} for ent in doc.ents]
    return entities

# Function to extract entities using fine-tuned SpaCy model
def extract_entities_spacy_finetuned(text):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model '{MODEL_PATH}' not found. Train the model first.")
        return []

    nlp = spacy.load(MODEL_PATH)
    doc = nlp(text)
    entities = [{"entity": ent.text, "type": ent.label_} for ent in doc.ents]
    return entities

# Streamlit app UI
st.title("Medical Entity Extraction Comparison")

# Input text field
user_input = st.text_area("Enter a medical text to extract entities:", height=250)

# Submit button to trigger entity extraction
if st.button("Submit"):
    if user_input:
        # Extract entities from all three approaches
        result_llm = extract_entities_llm(user_input)
        result_spacy_pretrained = extract_entities_spacy_pretrained(user_input)
        result_spacy_finetuned = extract_entities_spacy_finetuned(user_input)

        # Create data for the table: columns for each approach
        llm_entities = [(ent["entity"], ent["type"]) for ent in result_llm["entities"]] if "entities" in result_llm else []
        spacy_pretrained_entities = [(ent["entity"], ent["type"]) for ent in result_spacy_pretrained]
        spacy_finetuned_entities = [(ent["entity"], ent["type"]) for ent in result_spacy_finetuned]

        # Prepare a DataFrame to display as a table
        max_len = max(len(llm_entities), len(spacy_pretrained_entities), len(spacy_finetuned_entities))

        # Ensure all columns have the same number of rows
        llm_entities += [("", "")] * (max_len - len(llm_entities))
        spacy_pretrained_entities += [("", "")] * (max_len - len(spacy_pretrained_entities))
        spacy_finetuned_entities += [("", "")] * (max_len - len(spacy_finetuned_entities))

        # Create a DataFrame
        df = pd.DataFrame({
            "LLM Extracted Entity": [ent[0] for ent in llm_entities],
            "LLM Entity Type": [ent[1] for ent in llm_entities],
            "Pre-trained SpaCy Entity": [ent[0] for ent in spacy_pretrained_entities],
            "Pre-trained SpaCy Entity Type": [ent[1] for ent in spacy_pretrained_entities],
            "Fine-tuned SpaCy Entity": [ent[0] for ent in spacy_finetuned_entities],
            "Fine-tuned SpaCy Entity Type": [ent[1] for ent in spacy_finetuned_entities]
        })

        # Display the table
        st.subheader("Comparison of Extracted Entities and Their Types from Different Approaches")
        st.table(df)
    else:
        st.warning("Please enter some text.")
