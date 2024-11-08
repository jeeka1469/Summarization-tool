import streamlit as st
from transformers import pipeline
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK data is available
nltk.download('punkt')

# Summarizer class to handle extractive and abstractive summaries
class Summarizer:
    def __init__(self):
        self.extractor = TfidfVectorizer()
        self.abstractive_summarizer = pipeline("summarization", model="t5-small")

    def extractive_summary(self, text, num_sentences=3):
        sentences = nltk.sent_tokenize(text)
        tfidf_matrix = self.extractor.fit_transform(sentences)
        cosine_matrix = cosine_similarity(tfidf_matrix)
        summary_indices = cosine_matrix.sum(axis=1).argsort()[-num_sentences:]
        extractive_summary = " ".join([sentences[i] for i in sorted(summary_indices)])
        return extractive_summary

    def abstractive_summary(self, extractive_summary, max_length):
        abstractive_summary = self.abstractive_summarizer(
            extractive_summary,
            max_length=max_length,
            min_length=max_length - 20,
            do_sample=True,
            temperature=1.2,
            top_k=50,
            top_p=0.95
        )[0]['summary_text']
        return abstractive_summary

# Initialize the Summarizer
summarizer = Summarizer()

# Streamlit interface code
st.title("Text Summarizer")
st.write("This application generates both extractive and abstractive summaries of the input text.")

# Input text area
input_text = st.text_area("Input Text", height=200, placeholder="Enter the text you want to summarize...")

# Line counts for extractive and abstractive summaries
extractive_lines = st.number_input("Extractive Summary Lines", min_value=1, max_value=10, value=3)
abstractive_lines = st.number_input("Abstractive Summary Lines (approximate)", min_value=1, max_value=10, value=3)

# Generate summary button
if st.button("Generate Summaries"):
    if not input_text.strip():
        st.warning("Please enter text to summarize.")
    else:
        try:
            # Generate extractive summary
            extractive_summary = summarizer.extractive_summary(input_text, num_sentences=extractive_lines)
            st.subheader("Extractive Summary")
            st.write(extractive_summary)

            # Generate abstractive summary if extractive summary has sufficient content
            max_length_abstractive = abstractive_lines * 20  # Approximate max length per line
            if len(extractive_summary.split()) > 5:
                abstractive_summary = summarizer.abstractive_summary(extractive_summary, max_length=max_length_abstractive)

                # Similarity check between summaries
                if abstractive_summary.strip() == extractive_summary.strip():
                    abstractive_summary = "The abstractive summary is too similar to the extractive summary."
                
                st.subheader("Abstractive Summary")
                st.write(abstractive_summary)
            else:
                st.subheader("Abstractive Summary")
                st.write("Input text is too short for meaningful abstractive summarization.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Help section
with st.expander("Help"):
    st.write("""
    **To use this summarization tool:**
    
    1. Enter the text you want to summarize in the 'Input Text' area.
    2. Specify the number of lines for extractive and abstractive summaries.
    3. Click on 'Generate Summaries'.
    4. The summaries will appear below, which you can copy if needed.
    """)
