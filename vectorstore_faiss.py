from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

import resume_utils as parser



class vector_store():

    def __init__(self):
        '''
        Initialize the sentence transformer model
        '''
        self.st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def create_vectorstore(self, resume_text, jd_text):
        '''
        Embed each chunk from strings, normalize the embeddings 
        and build FAISS index 
        '''
        #split text to overlapping chunks 
        resume_chunks = [{"text" : chunk, "source" : "resume"} for chunk in parser.split_text(resume_text)]
        jd_chunks = [{"text" : chunk, "source" : "job"} for chunk in parser.split_text(jd_text)]

        #embed each chunk
        self.text_chunks = resume_chunks + jd_chunks
        self.texts = [chunk["text"] for chunk in self.text_chunks]
        embeddings = self.st_model.encode(self.texts)

        #Build FAISS index
        faiss.normalize_L2(embeddings)
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1]) #384 for Mini LM
        self.faiss_index.add(embeddings)


    def query_vectorstore(self, query, top_k=3):
        '''
        Embed the query from the user and return similarity scores
        '''
        #Embed query
        q_vector = self.st_model.encode([query])
        faiss.normalize_L2(q_vector)

        #Search FAISS
        distance_array, index_array = self.faiss_index.search(q_vector, k=top_k)
        results = []
        for i, score in zip(index_array[0], distance_array[0]):
            chunk = self.text_chunks[i]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": round(float(score), 3)  # Convert from np.float32
            })
        return results


if __name__ == "__main__":
    with open("fakepath\xyz.pdf", "rb") as file_name:
        r_text = parser.extract_text(file_name)
    with open("fakepath\sample.pdf", "rb") as f_name:
        j_text = parser.extract_text(f_name)
    
    vs = vector_store()
    vs.create_vectorstore(resume_text=r_text, jd_text=j_text)
    print(vs.query_vectorstore("RTOS"))