import re

def is_comparative_query(user_query):
    query = user_query.lower()
    keywords = [
        "missing", "compare", "not in my resume", "lacking",
        "gaps", "not mentioned", "difference", "absent"
    ]
    return any(term in query for term in keywords)

def build_prompt(user_query, resume_chunks, jd_chunks=None):
    r_text = "\n".join(resume_chunks)

    if is_comparative_query(user_query=user_query) and jd_chunks:
        jd_text = "\n".join(jd_chunks)
        return f"""
        Compare the user's resume and job description below:

        Resume:
        {r_text}

        Job Description:
        {jd_text}

        Question:
        {user_query}

        Instruction:
        List specific skills present in job description but missing in resume.
        """
    else:
        return f"""
        You are an assistant evaluating user's resume.

        Here is the resume context:
        {r_text}

        Question:
        {user_query}

        Instruction:
        Answer using the resume above. If something is not explicitly mentioned, say so clearly
        """