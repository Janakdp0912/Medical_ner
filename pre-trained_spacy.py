import spacy
import streamlit as st

# Load the pre-trained model
nlp = spacy.load("en_core_web_sm")

# Streamlit app layout
st.title("Named Entity Recognition (NER) with SpaCy pre trained model")
st.write(
    """
    This app uses a pre-trained SpaCy model to extract entities from text.
    Enter a sentence below, and the model will identify entities such as medical conditions, treatments, and more.
    """
)

# Input text box for the user to enter a text
user_input = st.text_area("the patient was diagnosed with Type 2 Diabetes Mellitus and prescribed Metformin. He also has a history of hypertension and chronic kidney disease")

# Perform NER on user input
doc = nlp(user_input)

# Display results
if user_input:
    st.subheader("Entities from the text:")
    for ent in doc.ents:
        st.write(f"{ent.text} - {ent.label_}")
else:
    st.write("Please enter text to analyze.")

