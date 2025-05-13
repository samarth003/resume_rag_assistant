import re, textwrap

MAX_CHUNKS = 4000

def build_prompt(user_query, resume_chunks, jd_chunks=None):
    resume_chunks = [
        "Developed embedded software for ARM Cortex-M microcontrollers.",
        "Worked with C and bare-metal environments for performance-critical applications."
        ]
    jd_chunks = [
        "Looking for someone with experience in RTOS, embedded Linux, and ARM-based firmware development."
        ]
    if not resume_chunks:
        return f"Resume content is empty. Cannot generate an answer."
    
    r_text = "\n".join(resume_chunks[:MAX_CHUNKS])
    jd_text = "\n".join(jd_chunks[:MAX_CHUNKS]) if jd_chunks else ""

    return textwrap.dedent(f"""
                           
    You are an AI assistant evaluating a candidate's resume in relation to a job description.
    Respond using this structure:
    - Relevant mentions in JD: [Yes/No + supporting lines if Yes]
    - Relevant mentions in Resume: [Yes/No + supporting lines if Yes]
    - Missing or mismatched points (if any): [list or state 'None']
    - Final Insight: [Short summary answering the user’s question clearly]

    Important:
    - Only use information explicitly provided.
    - Do not assume skills or experience unless clearly stated.
    
    Resume Context:
    {r_text}

    Job Description Context:
    {jd_text if jd_text else "Not provided."}

    Question:
    {user_query}

    """)