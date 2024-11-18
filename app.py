from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

app = Flask(__name__)

# Load your Excel or CSV file into a Pandas DataFrame
df = pd.read_csv("./data.csv")  # Replace with your file path

# Convert the DataFrame into a list of strings (one per row, formatted)
documents = df.to_dict(orient="records")  # List of dictionaries (rows)

# Define the LLaMa API call (or another language model)
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
texts = [str(doc) for doc in documents]  # Convert rows to text
retriever = FAISS.from_texts(
    texts,
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
).as_retriever(k=5)

# Define the FAQ prompt template
faq_template = """
You are a chat agent for my business. Help answer questions based on the following data:

<data>
{context}
</data>
"""

faq_prompt = ChatPromptTemplate.from_messages([
    ("system", faq_template),
    MessagesPlaceholder("messages")
])

# Create a document and retrieval chain
document_chain = LLMChain(llm=LLaMa(), prompt=faq_prompt)

# Function to parse the retriever input
def parse_retriever_input(params):
    return params["messages"][-1].content

# Define the retrieval chain
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
    
    answer_res = response['answer']
    
    if 'text' in answer_res:
        answer = answer_res['text']
    else:
        answer = "Sorry, I could not find an answer."

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True)
