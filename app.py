import gradio as gr
import vectorstore_faiss as vsf
import resume_utils as parser

class gradio_app():
    def __init__(self):
        self.vs = vsf.vector_store()
    
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
    
    def ask_question(self, user_query):
        '''
        User queries and search operation is triggered 
        returning with the formatted output
        user_query : user's related career question
        '''
        results = self.vs.query_vectorstore(query=user_query)
        formatted = "\n\n".join([
            f"**Source:** {r['source'].upper()}\n**Score:** {r['score']}\n{r['text']}" 
            for r in results
        ])
        return formatted
    
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
            fn=self.ask_question,
            inputs=gr.Textbox(label="Ask a Question", placeholder="e.g., What skills am I missing?"),
            outputs="text",
            title="Step 2: Ask Career Questions",
            description="Query your resume and job description for insights.",
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