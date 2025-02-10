import os
import json
import spacy
import streamlit as st
from groq import Groq

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
        st.subheader("Results from LLM:")
        with st.spinner("Extracting entities using LLM..."):
            try:
                result_llm = extract_entities_llm(user_input)
                st.json(result_llm)  # Display LLM extracted entities
            except Exception as e:
                st.error(f"Error occurred with LLM: {e}")

        st.subheader("Results from Pre-trained SpaCy Model:")
        with st.spinner("Extracting entities using pre-trained SpaCy model..."):
            try:
                result_spacy_pretrained = extract_entities_spacy_pretrained(user_input)
                st.write(result_spacy_pretrained)  # Display Pre-trained SpaCy extracted entities
            except Exception as e:
                st.error(f"Error occurred with Pre-trained SpaCy: {e}")

        st.subheader("Results from Fine-tuned SpaCy Model:")
        with st.spinner("Extracting entities using fine-tuned SpaCy model..."):
            try:
                result_spacy_finetuned = extract_entities_spacy_finetuned(user_input)
                st.write(result_spacy_finetuned)  # Display Fine-tuned SpaCy extracted entities
            except Exception as e:
                st.error(f"Error occurred with Fine-tuned SpaCy: {e}")

    else:
        st.warning("Please enter some text.")
