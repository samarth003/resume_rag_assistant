import pymupdf
import fitz

def extract_text(file_name):
    '''
    Function to parse the input file (.pdf)
    outputs string text
    '''
    #file parser - PDF
    text = ""
    with fitz.open(filename=file_name) as document:
        for page in document:
            text += page.get_text()
    return text.strip()

def split_text(extracted_text, chunk_size=800, overlap=100):
    '''
    Function to split the extracted string into overlapping chunks
    outputs list of strings
    '''
    text_chunks = []
    start = 0
    text_len = len(extracted_text)

    while start < text_len:
        end = start + chunk_size
        chunk = extracted_text[start:end]
        text_chunks.append(chunk)
        start += chunk_size - overlap

    return text_chunks
