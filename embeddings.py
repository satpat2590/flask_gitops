import chromadb
import openai 
import tiktoken
import PyPDF2 
import torch 
from transformers import BertTokenizer, BertModel
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import tempfile
from pytesseract import image_to_string
import os  

mpath = "learnings/neuronspaper.pdf"

print(os.uname())

### OPEN TEMP FILE (tempfile) -> TURN PDFs TO IMAGES (pdf2image) -> TURN IMAGES TO TEXT (pytesseract)
texts = [] 

def pdf_to_text(file_path):
    extracted_text = ""
    with tempfile.TemporaryDirectory() as path: 
        images_from_path = convert_from_path(file_path, output_folder=path)
        for image in images_from_path:
            extracted_text += image_to_string(image)
    texts.append(extracted_text)
    print(extracted_text)

def all_pdf_to_text():
    file_list = [(x, y, z) for x, y, z in os.walk('./conversions')]
    for idx, files in enumerate(file_list):
        if idx < 10:
            for file in files[2]:
                pdf_to_text(os.path.join(files[0], file))
        else:
            break 

all_pdf_to_text()

print(len(texts))

### INITIALIZE THE TIKTOKEN TOKEN ENCODER
encoding = tiktoken.get_encoding("cl100k_base")

### ENCODING EXAMPLE
#tokens = encoding.encode("tiktoken makes my cum nice and wet")
#print(tokens, len(tokens))
#print([encoding.decode_single_token_bytes(x) for x in tokens])


### USING PyPDF2 TO OPEN PDFs FROM THE ./learnings FOLDER
reader = PyPDF2.PdfReader(mpath)
pdf_to_embeddings = ""

for page in reader.pages:
    pdf_to_embeddings += page.extract_text()

#print(pdf_to_embeddings)

### TURNING STRINGS INTO TOKENS USING TIKTOKEN
tokenize = encoding.encode(pdf_to_embeddings)
decoded = [encoding.decode_single_token_bytes(x) for x in tokenize]

### GET OPENAI EMBEDDINGS
def get_embeddings(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

### INITIALIZE THE TIKTOKEN TOKEN ENCODER

### BERT TOKENIZER & EMBEDDDING + TORCH TENSOR PRACTICE
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokens created from using the 'tokenize' method, which simply creates the word-based tokens
tokens = tokenizer.tokenize("This is an example of a tokenized text")
# Token IDs created from using the 'encode' method, which turns word-based tokens into token IDs 
tokens2 = tokenizer.encode("This is an example of a tokenized text")
# Instantiate the BERT model from the BertModel module within transformers
model = BertModel.from_pretrained("bert-base-uncased")
example_token_id = tokenizer.convert_tokens_to_ids(['example', 'testing'])
example_embedding = model.embeddings.word_embeddings(torch.tensor(example_token_id))

cos = torch.nn.CosineSimilarity(dim=1)
#print(example_embedding)
