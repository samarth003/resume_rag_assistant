import re, textwrap

def is_comparative_query(user_query):
    query = user_query.lower()
    keywords = [
        "missing", "compare", "not in my resume", "lacking",
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
    r_text = "\n".join(resume_chunks)

    if is_comparative_query(user_query=user_query) and jd_chunks:
        jd_text = "\n".join(jd_chunks)
        return textwrap.dedent(f"""
        Compare the user's resume and job description below:

        Resume:
        {r_text}

        Job Description:
        {jd_text}

        Question:
        {user_query}

        Instruction:
        List specific skills present in job description but missing in resume.
        """)
    else:
        return textwrap.dedent(f"""
        You are an assistant evaluating user's resume.

        Here is the resume context:
        {r_text}

        Question:
        {user_query}

        Instruction:
        Only answer using the resume above. If something is not explicitly mentioned, say so clearly and do not assume.
        """)