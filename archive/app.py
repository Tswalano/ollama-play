from flask import Flask, request, jsonify, render_template
import requests
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import HumanMessage

app = Flask(__name__)

# Define the documents to upsert (as a list of strings)
documents = [
    "What is your return policy? Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with the receipt. Returns are processed within 7 business days.",
    "How long does shipping take? Standard shipping takes 5-7 business days. Expedited shipping options are available for faster delivery. International shipping times may vary depending on location.",
    "What payment methods do you accept? We accept major credit cards (Visa, MasterCard, American Express), PayPal, and Apple Pay. Payment can be made securely on our website during checkout.",
    "Can I change my order after it has been placed? Once an order is placed, it is processed immediately. However, you may be able to cancel or modify your order within 30 minutes of purchase by contacting customer support.",
    "Do you offer gift wrapping? Yes, we offer gift wrapping for an additional fee. You can select this option during checkout."
]

# Define the LLaMa API call
def call_llama(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        headers=headers,
        json=payload
    )
    return response.json()["response"]

# Create a custom LLaMa class
class LLaMa(LLM):
    def _call(self, prompt, **kwargs):
        return call_llama(prompt)

    @property
    def _llm_type(self):
        return "llama-3.1-8b"

# Extract document texts for FAISS retrieval
texts = documents  # Now using the plain list of strings
retriever = FAISS.from_texts(
    texts,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
).as_retriever(k=5)

# Define the FAQ prompt template
faq_template = """
You are a chat agent for my E-Commerce Company. As a chat agent, it is your duty to help the human with their inquiry and make them a happy customer.

Help them, using the following context:
<context>
{context}
</context>
"""

faq_prompt = ChatPromptTemplate.from_messages([
    ("system", faq_template),
    MessagesPlaceholder("messages")
])

# Create the document and retrieval chains
document_chain = create_stuff_documents_chain(LLaMa(), faq_prompt)

def parse_retriever_input(params):
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever
).assign(answer=document_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")

    # Handle the chat query through the retrieval chain
    response = retrieval_chain.invoke({
        "messages": [HumanMessage(user_query)]
    })

    # Extract the answer from the response
    answer = response['answer']

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
