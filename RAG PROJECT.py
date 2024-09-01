#!/usr/bin/env python
# coding: utf-8

# # 1 - ) Getting pdfs

# In[1]:


book1 = "16_EBOOK-7th_ed_software_engineering_a_practitioners_approach_by_roger_s._pressman_.pdf"
book2 = "218KİTABI.pdf"
book3 = "a_taylor_sql_for_dummies_2003.pdf"
book4 = "Agarwal and Lang (2005) Foundations of Analog and Digital.pdf"
book5 = "AI_Russell_Norvig.pdf"
book6 = "Alan_Beaulieu-Learning_SQL-EN.pdf"
book7 = "Andrew S. Tanenbaum - Computer Networks.pdf"
book8 = "Andrew S. Tanenbaum - Structured Computer Organization.pdf"
book9 = "andrew-ng-machine-learning-yearning.pdf"
book10 = "Computer Architecture A Quantitative Approach (5th edition).pdf"
book11 = "Computer Science- An Overview (12th Global Edition).pdf"
book12 = "ComputerNetworking.pdf"
book13 = "concepts.pdf"
book14 = "ConcreteAbstractions.pdf"
book15 = "Data Structures and Abstractions with Java 4th Edition.pdf"
book16 = "Elements_of_Theory_of_Computation_2ed_Lewis_Papadimitriou.pdf"
book17 = "essproglan.pdf"
book18 = "Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf"
book19 = "Information-Systems-A-Manager039s-Guide-to-Harnessing-Technology-1538497250._print.pdf"
book20 = "introduction-to-programming-in-java-an-interdisciplinary-approach-second-edition-9780134512396-0134512391_compress.pdf"
book21 = "Logic_and_Computer_Design_Fundamentals_-_4th_International_Edition.pdf"
book22 = "probability-and-statistics-for-computer-scientists-thirdnbsped-9781138044487-1138044482_compress.pdf"
book23 = "programming-language-pragmatics-2ed.pdf"
book24 = "Q6D2a4_Absolute_Java_(6th_edition_).pdf"
book25 = "Robert W. Sebesta - Concepts of Programming Languages-Pearson (2015).pdf"
book26 = "rosen_discrete_mathematics_and_its_applications_7th_edition.pdf"
book27 = "Sauer - Numerical Analysis 2e.pdf"
book28 = "Software-Engineering-9th-Edition-by-Ian-Sommerville.pdf"
book29 = "Starting-Out-With-Java.pdf"


# In[2]:


books = [book1,book2,book3,book4,book5,book6,book7,book8,book9,book10,book11,book12,book13,book14,book15,book16,book17,book18,
         book19,book20,book21,book22,book23,book24,book25,book26,book27,book28,book29]


# In[3]:


data = []


# In[ ]:





# In[ ]:





# In[4]:


import fitz
from tqdm.auto import tqdm
import re
import pandas as pd

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n"," ").strip()
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'\s+',' ',cleaned_text)
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc= fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_texts.append({"page_number": page_number,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text)/4,
                                "text": text})
        pages_and_texts_df = pd.DataFrame(pages_and_texts)
        
    return pages_and_texts_df

def open_and_read_pdf_list(pdf_list: list) -> list[dict]:
    for pdf in pdf_list:
        book = open_and_read_pdf(pdf_path = pdf)
        data.append(book)
pages_and_texts = open_and_read_pdf_list(pdf_list = books)


# In[5]:


described_data = []
mean_data = []
for i in range(len(data)):
    described_data.append(pd.DataFrame(data[i]["page_token_count"]).describe())


# In[6]:


mean_data = []
for i in range(len(data)):
    mean_data.append(data[i]["page_token_count"].mean())


# In[7]:


from spacy.lang.en import English

nlp = English()

# Add a sentencizer pipeline, see https://spacy.io/api/sentencizer 
nlp.add_pipe("sentencizer")

# Create document instance as an example
doc = nlp("This is a sentence. This another sentence. I like elephants.")
assert len(list(doc.sents)) == 3

# Print out our sentences split
list(doc.sents)


# In[8]:


for df in tqdm(data):
    df["sentences"] = df["text"].apply(lambda text: list(nlp(text).sents))
    df["sentences"] = df["sentences"].apply(lambda sentences: [str(sentence) for sentence in sentences])
    df["page_sentence_count_spacy"] = df["sentences"].apply(len)


# In[9]:


df = pd.DataFrame(data)
df.describe().round(2)


# In[10]:


data[0].describe()


# In[11]:


# Define split size to turn groups of sentences into chunks
num_sentence_chunk_size = 10

# Create a function to split lists of texts recursively into chunk size
# e.g. [20] -> [10, 10] or [25] -> [10, 10, 5]
def split_list(input_list: list[str],
               slice_size: int=num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

test_list = list(range(25))
split_list(test_list)


# In[12]:


# Loop through pages and texts and split sentences into chunks
for df in tqdm(data):
    df["sentence_chunks"] = df["sentences"].apply(lambda sentences: split_list(input_list=sentences, slice_size=num_sentence_chunk_size))
    df["num_chunks"] = df["sentence_chunks"].apply(len)


# In[13]:


import random
random.sample(data,k=1)


# In[14]:


# Split each chunk into its own item
chunks = []
for df in tqdm(data):
    for index, row in df.iterrows():
        for sentence_chunk in row["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = row["page_number"]

            # Join the sentences together into a paragraph-like structure
            joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" => ". A"

            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get some stats on our chunks
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4  # 1 token = ~4 chars

            chunks.append(chunk_dict)

print(len(chunks))


# In[15]:


random.sample(chunks, k=1)


# In[16]:


df = pd.DataFrame(chunks)
df.describe().round(2)


# In[17]:


df.head()


# In[18]:


# Show random chunks with under 15 tokens in length
min_token_length = 15
for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
    print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["sentence_chunk"]}')


# In[19]:


# Filter our DataFrame for rows with under 15 tokens
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
pages_and_chunks_over_min_token_len[:5]


# In[20]:


random.sample(pages_and_chunks_over_min_token_len, k=1)


# In[22]:


from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")

# Create a list of sentences
sentences = ["The Sentence Transformer library provides an easy way to create embeddings.",
             "Sentences can be embedded one by one or in a list.",
             "I like horses!"]

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# See the embeddings
for sentence, embedding in embeddings_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}")
    print("")


# In[23]:


embeddings[0].shape


# In[24]:


get_ipython().run_cell_magic('time', '', '\nfor item in tqdm(pages_and_chunks_over_min_token_len):\n    item["embedding"] = embedding_model.encode(item["sentence_chunk"])\n')


# In[36]:


pages_and_chunks_over_min_token_len


# In[21]:


import torch


# In[38]:


pip install faiss-cpu 


# In[22]:


text_array = [item['sentence_chunk']for item in pages_and_chunks_over_min_token_len]


# In[24]:


import sqlite3

# Bağlantı oluştur ve tabloyu oluştur
conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS text_data (item TEXT)''')

# Veriyi ekle
c.executemany('INSERT INTO text_data (item) VALUES (?)', [(item,) for item in text_array])

conn.commit()
conn.close()


# In[34]:


import numpy as np


# In[52]:


embedings_array = [item["embedding"]for item in pages_and_chunks_over_min_token_len]

embeddings_list = list(embedings_array)
embeddings = np.array(embeddings_list).astype('float32')


# In[54]:


print(embeddings.shape)


# In[2]:


import faiss


# In[57]:


d = embeddings.shape[1]

index = faiss.IndexFlatL2(d)

index.add(embeddings)

faiss.write_index(index, "CS_RAG.index")


# In[42]:


pages_and_chunks_over_min_token_len


# In[1]:


import faiss
# İndeksi yükleyin
index = faiss.read_index('CS_RAG.index')

# İndeksin bazı özelliklerini kontrol edin
print(f"İndeks türü: {type(index)}")
print(f"Veri boyutu: {index.ntotal}")


# In[2]:


import numpy as np
import pandas as pd


# In[4]:


from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")

querry = ["Give me brief explenation about float data type."]

# Sentences are encoded/embedded by calling model.encode()
querry_embeddings = embedding_model.encode(querry)


# In[6]:


# Bağlantı oluştur ve veriyi oku
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

c.execute('SELECT item FROM text_data')
long_list = [row[0] for row in c.fetchall()]

conn.close()


# In[7]:


k = 5
D , I = index.search(querry_embeddings,k)
answer = []
for i in I[0]:
    answer.append(long_list[i])


# In[8]:


print(answer)


# In[9]:


print(I)


# In[27]:


len(long_list)


# In[9]:


pip install openai


# In[4]:


import openai

openai.api_key = ''



# Dosyayı yükleme
response = openai.File.create(
  file=open("fine_tune.jsonl", "rb"),
  purpose="fine-tune"
)

# Yanıtı yazdırma
print(response)


# In[5]:


file_id = response["id"]


# In[6]:


file_id


# In[9]:





# In[1]:


from openai import OpenAI
import openai

client = OpenAI(api_key = '')

client.files.create(
  file=open("fine_tune.jsonl", "rb"),
  purpose="fine-tune"
)


# In[2]:


openai.api_key = ''

client.fine_tuning.jobs.create(
  training_file="", 
  model="gpt-3.5-turbo"
)


# In[ ]:


pip uninstall openai


# In[ ]:


pip install openai


# In[3]:


import time

fine_tune_job_id = ""

while True:
    job_status = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
    status = job_status.status
    print(f"Fine-tuning job status: {status}")

    if status == 'succeeded':
        print("Fine-tuning job completed successfully!")
        fine_tuned_model_id = job_status.fine_tuned_model
        print(f"Fine-tuned model ID: {fine_tuned_model_id}")
        break
    elif status in ['failed', 'cancelled']:
        print(f"Fine-tuning job failed or was cancelled: {job_status.error}")
        break

    time.sleep(60)  # Check status every 60 seconds


# In[13]:


from openai import OpenAI
import openai
client = OpenAI(api_key = '')
completion = client.chat.completions.create(
  model="",
  messages=[
    {"role": "system", "content": "You are a chatbot that answer's question about Computer Science field accourding to given relevant data."},
    {"role": "user", "content": """real data type the real data type gives you a single-precision floating-point number, the precision of which depends on the implementation. in general, the hardware you’re using determines precision. a 64-bit machine, for example, gives you more precision than does a 32-bit machine. a floating-point number is a number that contains a decimal point. the decimal point “floats” or appears in different locations in the number, depending on the number’s value. 3.1, 3.14, and 3.14159 are examples of floating-point numbers. double precision data type the double precision data type gives you a double-precision floating- point number, the precision of which again depends on the implementation. surprisingly, the meaning of the word double also depends on the implemen- tation. double-precision arithmetic is primarily employed by scientific users. different scientific disciplines have different needs in the area of precision.', 'of course, neither of these numbers can be precisely represented in any finite amount of computer memory. on most computers, \xad floating-\u200b \xad point numbers are stored in binary, which exacerbates the problem. for example, even the value 0.1 in decimal cannot be represented by a finite number of binary digits.1 another problem with \xad floating-\u200b \xad point types is the loss of accuracy through arithmetic operations. for more information on the problems of \xad floating-\u200b \xad point notation, see any book on numerical analysis. \xad floating-\u200b \xad point values are represented as fractions and exponents, a form that is borrowed from scientific notation. older computers used a variety of different representations for \xad floating-\u200b \xad point values. however, most newer machines use the ieee \xad floating-\u200b \xad point standard 754 format. language imple- mentors use whatever representation is supported by the hardware. most lan- guages include two \xad floating-\u200b \xad point types, often called float and double. the float type is the standard size, usually stored in four bytes of memory.', 'many programming languages traditionally used two types for floating-point numbers. one type used less storage and was very imprecise (that is, it did not allow very many significant digits). the second type used double the amount of storage and so could be much more precise; it also allowed numbers that were larger (although programmers tend to care more about precision than about size). the kind of numbers that used twice as much storage were called double precision numbers; those that used less storage were called single precision. following this tradition, the type that (more or less) corresponds to this double precision type in java was named double in java. the type that corresponds to single precision in java was called float. ( actually, the type name double was inherited from c++, but this explanation applies to why the type was named double in c++, and so ultimately it is the explanation of why the type is called double in java.) quotes mixing types', '474 chapter 11 abstract data types and encapsulation constructs \xad object-\u200b \xad oriented programming, which is described in chapter 12, is an outgrowth of the use of data abstraction in software development, and data abstraction is one of its fundamental components. 11.2.1 \xad floating-\u200b \xad point as an abstract data type the concept of an abstract data type, at least in terms of \xad built-\u200b \xad in types, is not a recent development. all \xad built-\u200b \xad in data types, even those of fortran i, are abstract data types, although they are rarely called that. for example, consider a \xad floating-\u200b \xad point data type. most programming languages include at least one of these. a \xad floating-\u200b \xad point type provides the means to create variables to store \xad floating-\u200b \xad point data and also provides a set of arithmetic operations for manipulating objects of the type. \xad floating-\u200b \xad point types in \xad high-\u200b \xad level languages employ a key concept in data abstraction: information hiding. the actual format of the \xad floating-\u200b \xad point data value in a memory cell is hidden from the user, and the only operations available are those provided by the language. the user is not allowed to create new operations on data of the type, except those that can be constructed using the \xad built-\u200b \xad in opera- tions. the user cannot directly manipulate the parts of the actual representation of values because that representation is hidden.', 'the double type is provided for situations where larger fractional parts and/or a larger range of exponents is needed. \xad double-\u200b \xad precision variables usually occupy twice as much storage as float variables and provide at least twice the number of bits of fraction. the collection of values that can be represented by a \xad floating-\u200b \xad point type is defined in terms of precision and range. precision is the accuracy of the frac- tional part of a value, measured as the number of bits. range is a combination of the range of fractions and, more important, the range of exponents. figure 6.1 shows the ieee \xad floating-\u200b \xad point standard 754 format for \xad single-\u200b \xad and \xad double-\u200b \xad precision representation (ieee, 1985). details of the ieee formats can be found in t anenbaum (2005). 1. 0.1 in decimal is 0.0001100110011... in binary.
     Question:Give me brief explenation about float data type."""}
  ]
)

print(completion.choices[0].message.content)


# In[1]:


# Bağlantı oluştur ve veriyi oku
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

c.execute('SELECT item FROM text_data')
long_list = [row[0] for row in c.fetchall()]

conn.close()


# In[2]:


import faiss
# İndeksi yükleyin
index = faiss.read_index('CS_RAG.index')


# In[5]:


from sentence_transformers import SentenceTransformer
from openai import OpenAI
import openai
def main(querry,k):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device="cpu")
    
    question = querry
    querry = [querry]

    # Sentences are encoded/embedded by calling model.encode()
    querry_embeddings = embedding_model.encode(querry)
    
    D , I = index.search(querry_embeddings,k)
    content = ""
    answer = []
    for i in I[0]:
        answer.append(long_list[i])
    
    for ans in answer:
        content += ans
        
    client = OpenAI(api_key = '')
    completion = client.chat.completions.create(
        model="",
        messages=[
            {"role": "system", "content": "You are a chatbot that answer's question about Computer Science field accourding to given relevant data."},
            {"role": "user", "content": f"""{content}
            Question:{question}"""}
          ]
        )

    print(completion.choices[0].message.content)


# In[9]:


main(""" Explain the differences between compiler and interpreter. Discuss the advantages and disadvantages of these two approaches.""",5)


# In[ ]:


question1 = "What is system call? Give me an example"
question2 = "What do cohesion and coupling mean in computer science? Which one is good in OOP? "
question3 = "Explain the differences between compiler and interpreter. Discuss the advantages and disadvantages of these two approaches."

