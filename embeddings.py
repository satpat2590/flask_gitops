import chromadb
import openai 
import tiktoken
import PyPDF2 
import torch 
from transformers import BertTokenizer, BertModel

encoding = tiktoken.get_encoding("cl100k_base")

### ENCODING EXAMPLE
#tokens = encoding.encode("tiktoken makes my cum nice and wet")
#print(tokens, len(tokens))
#print([encoding.decode_single_token_bytes(x) for x in tokens])


### READING PDF FROM THE ./learnings FOLDER
reader = PyPDF2.PdfReader("learnings/Implementation_Consultant.pdf")
pdf_to_embeddings = ""

for page in reader.pages:
    pdf_to_embeddings += page.extract_text()

### TURNING STRINGS INTO TOKENS
tokenize = encoding.encode(pdf_to_embeddings)
decoded = [encoding.decode_single_token_bytes(x) for x in tokenize]

### GET OPENAI EMBEDDINGS
def get_embeddings(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


### BERT TOKENIZER & TORCH EMBEDDING PRACTICE
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
print(example_embedding)
