import gradio as gr
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os, threading

import vectorstore_faiss as vsf
import resume_utils as parser
import promptbuilder as pb

INFERENCE_TEST = False

class gradio_app():
    def __init__(self):
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        self.vs = vsf.vector_store()
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token = hf_token)
        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
        if INFERENCE_TEST:
            try:
                print("Running model warm up test...")
                test_out = self.generator("Say hello.", max_new_tokens=10)
                print("Inference success", test_out[0]['generated_text'])
            except Exception as e:
                print(f"Inference failed: {e}")
    
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
    
    def safe_generate(self, user_prompt, timeout=120):
        result = {"done" : False, "value" : "Generating response..."}
        def run():
            try:
                print(f"[DEBUG] Calling LLM...")
                llm_response = self.generator(user_prompt, max_new_tokens=256,
                                              do_sample=True, temperature=0.7)
                print(f"[DEBUG] LLM responded", llm_response[0]['generated_text'][:300])

                result["value"] = llm_response[0]['generated_text']
            except Exception as e:
                result["value"] = f"Error during generation : {e}"
            result["done"] = True

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=timeout)

        print(f"[DEBUG] Thread Finished: {result['done']}")
        print(f"[DEBUG] Result value : {result['value'][:200]}")

        if not result["done"]:
            return "Model took too long to respond. Try again."
        return result["value"]
    
    def generate_answer(self, user_query):
        '''
        Retrieve top chunks from vectorstore
        Build contextual prompt
        Runs through LLM to get response 
        '''
        if INFERENCE_TEST:
            try:
                response = self.generator("What is AI?", max_new_tokens=30)
                return response[0]['generated_text']
            except Exception as e:
                return f"Error during direct inference: {e}"
        else:
            if not hasattr(self.vs, "faiss_index"):
                return "Please upload resume and JD first."
            top_chunks = self.vs.query_vectorstore(query=user_query)
            if not top_chunks:
                return "No relevant context found to answer your question"
            
            r_context = [chunk["text"] for chunk in top_chunks if chunk["source"]=="resume"]
            jd_context = [chunk["text"] for chunk in top_chunks if chunk["source"]=="job"]

            prompt = pb.build_prompt(user_query=user_query, resume_chunks=r_context, jd_chunks=jd_context)
            
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
            ],
            cache_examples=False
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