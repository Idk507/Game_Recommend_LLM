from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS 
from langchain.llms import CTransformers 
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain 
import os 
import sys 
import tempfile
import re 


loader = CSVLoader(file_path='games.csv',encoding="utf-8",csv_args={'delimiter':','})

data = loader.load()

#print(data)

#split the text into chunks 

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

#print(len(text_chunks))

#embeddings -Sentence Transformers 

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

#Converting the text chunks into embeddings and saving the embeddings into FAISS knowledge base 

DB_FAISS = "vector_Store/db_faiss"

docsearch = FAISS.from_documents(text_chunks,embeddings)

docsearch.save_local(DB_FAISS)


query = "What is the games are available?"

docs = docsearch.similarity_search(query,k=3)

print("Result",clean_text(docs))


#defining LLAMA2 model 

llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
                    model_type = "llama",
                    max_new_tokens=512,
                    temperature = 0.1)

qa = ConversationalRetrievalChain.from_llm(llm,retriever=docsearch.as_retriever())

def clean_text(text):
    """Clean and format text to be more readable."""
    text = re.sub(r"\\n", "\n", text)
    text = re.sub(r"\\'", "'", text)
    text = re.sub(r'["\[\]]', '', text)  # Remove brackets and quotes
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Conversation loop
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = qa({"question": query, "chat_history": chat_history})
    answer = clean_text(response['answer'])
    chat_history.append((query, answer))
    print("Bot:", answer)
    

