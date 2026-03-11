#!/usr/bin/env python
# coding: utf-8

# # Chapter 2 Working with Data

# In[5]:


from importlib.metadata import version

pkg = ['tiktoken','torch']
for p in pkg:
    print(f"{p} version: {version(p)}")


# In[11]:


import os 
import requests

if not os.path.exists("the-verdict.txt"):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-verdict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)


# In[13]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])


# In[23]:


import re
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# In[25]:


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)


# In[27]:


vocab = {token:integer for integer,token in enumerate(all_words)}


# In[29]:


class TokenizerV1:
    def __init__ (self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    


# In[33]:


tokenizer = TokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


# In[37]:


decodedText = tokenizer.decode(ids)
print(decodedText)


# In[39]:


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}


# In[41]:


for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# In[43]:


#manages edge cases when the vocabulary doesnt include the word by using special tokens such as </unk>
class TokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


# ## DataLoaders

# In[80]:


from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

class GPTDataSet(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids= []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})

    # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0,len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1 : i+max_length +1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# In[82]:


def createDataLoader(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDataSet(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
    


# In[84]:


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# In[86]:


dataloader = createDataLoader(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)


# In[90]:


dataloader = createDataLoader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# In[106]:


input_ids = torch.tensor([2, 3, 5, 1])


# In[108]:


vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)


# In[110]:


count = 0
for i in embedding_layer.weight:
    count+=1
    print(count, i)


# In[112]:


embedding_layer(input_ids)


# In[116]:


#now we use the full vocabulary used by the gpt2 model and configure a 256 dimensional vector
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# In[124]:


max_length = 4
dataloader = createDataLoader(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)


# In[126]:


print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


# In[130]:


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)


# In[138]:


pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print(pos_embeddings.shape)


# In[142]:


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings)


# In[ ]:




