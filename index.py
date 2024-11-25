from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b")
llm.invoke("in 10 words describe why the skyb is blue")