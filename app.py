import gradio as gr
import vectorstore_faiss as vsf
import resume_utils as parser
from transformers import pipeline
import torch, os, threading

class gradio_app():
    def __init__(self):
        hf_token = os.environ.get("HF_Token")
        self.vs = vsf.vector_store()
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            use_auth_token=hf_token,
            device=0 if torch.cuda.is_available() else -1,
        )
    
    def upload_files(self, resume_file, jd_file):
        '''
        Upload Resume and paste job description text
        checks for any error during upload process
        resume_file : resume file input 
        jd_file     : job description text 
        '''
        try:
            r_text = parser.extract_text(resume_file)
            j_text = jd_file.strip()
            self.vs.create_vectorstore(resume_text=r_text, jd_text=j_text)
            return "Files processed. You can now ask related career questions!"
        except Exception as e:
            return f"Error during upload: {e}"
    
    def safe_generate(self, user_prompt, timeout=45):
        result = ["Generating response ..."]
        def run():
            llm_response = self.generator(user_prompt, max_new_tokens=256,
                                 do_sample=True, temperature=0.7)
            result[0] = llm_response[0]['generated_text']
        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return "Model took too long to respond. Try again."
        return result[0]
    
    def generate_answer(self, user_query):
        '''
        Retrieve top chunks from vectorstore
        Build contextual prompt
        Runs through LLM to get response 
        '''
        top_chunks = self.vs.query_vectorstore(query=user_query)
        if not top_chunks:
            return "No relevant context found to answer your question"
        
        context = "\n".join([chunk["text"] for chunk in top_chunks])

        prompt = f"""Answer the following question using the context below.

        Context:
        {context}


        Question:
        {user_query}

        """
        
        return self.safe_generate(user_prompt=prompt)
    
    def gradio_upload_IF(self):
        '''
        Interface for uploading files
        '''
        self.upload_interface = gr.Interface(
            fn=self.upload_files,
            inputs=[
                gr.File(label="Upload Resume (PDF)"),
                gr.Textbox(label="Paste Job Description text here ...")
            ],
            outputs=gr.Markdown(),
            title="Step 1: Upload Your Documents",
            description="Upload your Resume and paste Job Description text to begin.",
        )

    def gradio_query_IF(self):
        '''
        Interface for user queries
        '''
        self.query_interface = gr.Interface(
            fn=self.generate_answer,
            inputs=gr.Textbox(label="Ask a Question", placeholder="e.g., What skills am I missing?"),
            outputs="text",
            title="Step 2: Get Career Insights",
            description="Ask questions like skill gaps, resume tips or keyword matches.",
            examples=[
                ["What skills am I missing?"],
                ["Which job keywords are not in my resume?"],
                ["Suggest a better resume summary for this job."]
            ]
        )        

    def gradio_IF_launch(self):
        '''
        Application initialize and launch
        '''
        self.gradio_upload_IF()
        self.gradio_query_IF()
        app = gr.TabbedInterface([self.upload_interface, self.query_interface], ["Upload", "Ask"])
        app.launch(share=True)

if __name__ == "__main__":
    gr_app = gradio_app()
    gr_app.gradio_IF_launch()