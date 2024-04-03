prompt_template = """
The user will provide you with a list of symptoms, and issues they are facing and maybe a question. Use the symptoms to find the most relevant medical condition they are suffering from and provide a simple diagnosis and then answer the question based on that context.
Once you provide the diagnosis also suggest a treatment plan for the user to follow.
If you are unable to diagnose based on the symptoms, ask for more information to arrive at a more clearer diagnosis.
Use the following pieces of information to answer the user.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
# prompt_template = """
# Context: {context}
# Question: {question}

# Provide a response to the user's symptoms or questions below:
# """
