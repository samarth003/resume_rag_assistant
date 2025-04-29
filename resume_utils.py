import pymupdf
import fitz
'''
extract_text()
Function to parse the input file (.pdf)
outputs string text
'''
def extract_text(file_name):
    #file parser - PDF
    text = ""
    with fitz.open(stream=file_name.read(), filename="pdf") as document:
        for page in document:
            text += page.get_text()
    return text.strip()

'''
split_text()
Function to split the extracted string into overlapping chunks
outputs list of strings
'''
def split_text(extracted_text, chunk_size=800, overlap=100):

    text_chunks = []
    start = 0
    text_len = len(extracted_text)

    while start < text_len:
        end = start + chunk_size
        chunk = extracted_text[start:end]
        text_chunks.append(chunk)
        start += chunk_size - overlap

    return text_chunks


if __name__ == "__main__":
    with open("fakepath\xyz.pdf", "rb") as file_name:
        extracted_text = extract_text(file_name)
    print(extracted_text)
    message_chunks = split_text(extracted_text=extracted_text)
    print(message_chunks)