import spacy
from docx import Document

def extract_word_text(file_path):
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs if para.text.strip() != ""]

    return list_to_string(full_text)

def list_to_string(list):
    return " ".join(list)
