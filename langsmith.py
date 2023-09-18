import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader


converted_docs = TextLoader('./data/converted/hydrophobicity.txt').load()

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

texts = text_splitter.split_documents(converted_docs)

db = Chroma.from_documents(texts, OpenAIEmbeddings())

query = "What's the gist of hydrophobicity?"

docs = db.similarity_search(query)

print(docs[0].page_content)