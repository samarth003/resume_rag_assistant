import re, textwrap

MAX_CHUNKS = 4000

def is_comparative_query(user_query):
    query = user_query.lower()
    keywords = [
        "missing", "compare", "not in my resume", "lacking", "need",
        "gaps", "not mentioned", "difference", "absent", "require"
    ]
    return any(term in query for term in keywords)

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

    if is_comparative_query(user_query=user_query) and jd_chunks:
        jd_text = "\n".join(jd_chunks[:MAX_CHUNKS])
        return textwrap.dedent(f"""
        Given the resume and job description context, answer the following question below.

        Resume Context:
        {r_text}

        Job Description Context:
        {jd_text}

        Question:
        {user_query}

        Instruction:
        Answer should only be specific to the question asked. Do not assume.
        """)
    else:
        return textwrap.dedent(f"""
        You are an assistant evaluating user's resume. 
        Given the resume context, answer the following question below.

        Here is the resume context:
        {r_text}

        Question:
        {user_query}

        Instruction:
        Answer should only be specific to the question asked. Do not assume.
        """)