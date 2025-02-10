import os
import json
import streamlit as st
from groq import Groq

# Initialize Groq client
client = Groq(
    api_key="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"
)

def extract_entities(text):
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

# Streamlit app UI
st.title("Medical Entity Extraction using llm")
st.write(
    "Enter a medical text to extract relevant entities like diseases, conditions, medications, etc."
)

# Text input field
user_input = st.text_area("Input Text", height=250)

# Button to extract entities
if st.button("Extract Entities"):
    if user_input:
        with st.spinner("Extracting entities..."):
            try:
                result = extract_entities(user_input)
                st.subheader("Extracted Entities:")
                st.json(result)  # Display JSON response with entities
            except Exception as e:
                st.error(f"Error occurred: {e}")
    else:
        st.warning("Please enter some text.")