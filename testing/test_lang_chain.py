import langchain_community.llms
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
llm.invoke("why is the sky blue")