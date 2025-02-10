import os
import json
import spacy
import streamlit as st
from groq import Groq
import pandas as pd

# Constants
MODEL_PATH = "/home/atliq/Music/poc_spacy/fine_tuned_model"
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
        entities_llm = [ent["entity"] for ent in result_llm["entities"]] if "entities" in result_llm else []
        entities_spacy_pretrained = [ent["entity"] for ent in result_spacy_pretrained] 
        entities_spacy_finetuned = [ent["entity"] for ent in result_spacy_finetuned]

        # Prepare a DataFrame to display as a table
        max_len = max(len(entities_llm), len(entities_spacy_pretrained), len(entities_spacy_finetuned))
        
        # Ensure all columns have the same number of rows
        entities_llm += [""] * (max_len - len(entities_llm))
        entities_spacy_pretrained += [""] * (max_len - len(entities_spacy_pretrained))
        entities_spacy_finetuned += [""] * (max_len - len(entities_spacy_finetuned))

        # Create a DataFrame
        df = pd.DataFrame({
            "LLM Extracted Entities": entities_llm,
            "Pre-trained SpaCy Entities": entities_spacy_pretrained,
            "Fine-tuned SpaCy Entities": entities_spacy_finetuned
        })

        # Display the table
        st.subheader("Comparison of Extracted Entities from Different Approaches")
        st.table(df)
    else:
        st.warning("Please enter some text.")
