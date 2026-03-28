from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model=ChatGroq(model="llama-3.3-70b-versatile")

#Step2: Retrieve Docs

def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
You are an expert AI Lawyer. Use the context provided to answer the user's legal question in a clear, structured, and professional manner.

Format your response as follows:
- Start with a direct answer to the question
- List the specific articles or laws violated (if any), with their titles
- Explain WHY each article is violated with brief reasoning
- End with a brief conclusion

If the answer is not in the context, say: "I don't have enough information in the provided document to answer this question."

Question: {question}
Context: {context}
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

#question="If a government forbids the right to assemble peacefully which articles are violated and why?"
#retrieved_docs=retrieve_docs(question)
#print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))