from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
import streamlit as st
import tempfile

# Load model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def compute_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(sim_score * 100, 2)  # return percentage
