# prompt_template = """
# The user will provide you with a list of symptoms and issues they are facing. Use the symptoms to find the most relevant medical condition they are suffering from and provide a simple diagnosis. Additionally, suggest a treatment plan for the user to follow.
# If you are unable to diagnose based on the symptoms, ask for more information to arrive at a clearer diagnosis.

# Use the following pieces of information to answer the user:

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else
# """
prompt_template = """Assume the role of a diagnostic chatbot that will help patients to understand what they are suffering from using a list of symtpoms they are facing. 
If you are not able to arrive at a diagnosis, do not provide a diagnosis.

Question: {question}
Context: {context}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

