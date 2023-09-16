import openai 
import tiktoken
import PyPDF2 
import torch 
from transformers import BertTokenizer, BertModel
from pdf2image import convert_from_path
from PIL import Image
try: 
    import chromadb 
except Exception: 
    print("chromadb is not able to be configured on your machine...\n\n")
import tempfile
from pytesseract import image_to_string
import os  

mpath = "conversions/neuronspaper.pdf"

### os MODULE METHODS BEING RUN IN THIS PORTION
system_info = os.uname()

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
#tokens = encoding.encode("tiktoken makes my code nice and manageable")
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



class Embeddings():
    def __init__(self, path: str):
        self.embedder = BertModel.from_pretrained('bert-case-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-case-uncased')
        self.similarity_search = torch.nn.CosineSimilarity(dim=1) # 'dim' refers to either rows or columns (0 or 1 respectively)
        self.tiktokenizer = tiktoken.get_encoding('cl100k_base') # tokenizer using the 'tiktoken' module
        self.file_path = path
        self.texts = [] 


    def pdf_to_text(self, file_path):
        """
        Turning PDF -> Image -> Text using the following steps: 
            1. Create a temporary directory using the 'tempdir' module
            2. Use the 'convert_from_path' method in 'pdf2image' to turn specific PDF into 
                an image in a temporary directory 
            3. Use the 'pytesseract' module's 'image_to_string' method to turn the image to a string
            4. Append the newly extracted text from the OCR process to the self.texts array 
        
        :param file_path: The file in which you will be extracting PDFs from 
        :return: Returns a populated self.texts array, which holds PDF information in string format
        """
        extracted_text = ""
        with tempfile.TemporaryDirectory() as path: 
            images_from_path = convert_from_path(file_path, output_folder=path)
            for image in images_from_path:
                extracted_text += image_to_string(image)
        self.texts.append(extracted_text)
        print(extracted_text) 


    def all_pdf_to_text(self):
        """
        Open the directory (self.file_path) and walk through it to grab all PDFs and run the 
        self.pdf_to_text method on each of these PDFs. 

        :return: Once complete, the self.texts array will be fully populated with embeddings-capable text
            from the PDFs 
        """
        file_list = [(x, y, z) for x, y, z in os.walk(self.file_path)]
        for idx, files in enumerate(file_list):
            if idx < 10:
                for file in files[2]:
                    self.pdf_to_text(os.path.join(files[0], file))
            else:
                break 

    def get_embeddings(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


embeddings = Embeddings('./conversions')