# Using AI to Reduce the 6G Standards Barrier for African Contributors
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F769452%2Fb18d0513200d426e556b2b7b7c825981%2FRAG.png?generation=1695504022336680&alt=media"></img>

## Objective

Use Llama 2.0, Langchain and ChromaDB to create a Retrieval Augmented Generation (RAG) system. This will allow us to ask questions about our documents (that were not included in the training data), without fine-tunning the Large Language Model (LLM).
When using RAG, if you are given a question, you first do a retrieval step to fetch any relevant documents from a special database, a vector database where these documents were indexed. 

## Definitions

* LLM - Large Language Model  
* Llama 2.0 - LLM from Meta 
* Langchain - a framework designed to simplify the creation of applications using LLMs
* Vector database - a database that organizes data through high-dimmensional vectors  
* ChromaDB - vector database  
* RAG - Retrieval Augmented Generation (see below more details about RAGs)

## Model details

* **Model**: Llama 2  
* **Variation**: 7b-chat-hf  (7b: 7B dimm. hf: HuggingFace build)
* **Version**: V1  
* **Framework**: PyTorch  

LlaMA 2 model is pretrained and fine-tuned with 2 Trillion tokens and 7 to 70 Billion parameters which makes it one of the powerful open source models. It is a highly improvement over LlaMA 1 model.


## What is a Retrieval Augmented Generation (RAG) system?

Large Language Models (LLMs) are great at understanding context and answering NLP tasks like summarization or question answering. However, they tend to hallucinate when asked about information outside their training data. RAG mitigates this by combining LLMs with external data retrieval.

A RAG system has two main components:
- **Retriever**: Encodes and retrieves relevant parts of your data using text embeddings. A vector database, like **ChromaDB**, organizes this data. 
- **Generator**: The LLM that generates answers based on retrieved data.


In this project, we use a quantized Llama 2 model as the generator and ChromaDB as the vector database.

## Step-by-Step Workflow
- **Initialize Model and Tokenizer:** Load the Llama 2 model and tokenizer.
- **Define the Query Pipeline:** Set up a pipeline for text generation.
- **Data Ingestion and Splitting:** Load and split documents into smaller chunks.
- **Embedding Generation:** Generate embeddings using a sentence transformer.
- **Store in ChromaDB:** Store document embeddings in a vector database (ChromaDB).
- **RAG Query Pipeline:** Combine the retriever and Llama 2 generator using Langchain to handle queries.

## Installations, imports, and utilities

Before running the code, ensure you have Python 3.8+ installed, and it's recommended to set up a virtual environment. You can install the required packages with the following commands:

```python
!pip install transformers==4.33.0 accelerate==0.22.0 einops==0.6.1 langchain==0.0.300 xformers==0.0.21 bitsandbytes==0.41.1 sentence_transformers==2.2.2 chromadb==0.4.12
```

## References

- [Llama 2.0 Model Overview - Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-2/)
- [Langchain Documentation](https://langchain.readthedocs.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Transformers Documentation - HuggingFace](https://huggingface.co/docs/transformers/)

## Contributors

**TeamLead Name 1 – Emmanuel Othniel Eggah**

**Teammate Name 2 – Eniola Alao**

**Teammate Name 3 – Frank Chukwubuikem Ebeledike**

**Teammate Name 4- Aaron Emmanuel Enejo**

**Teammate Name 5- Victor Onah Chukwuebuka**

**Teammate Name 6- Adegoke Israel Adedolapo**
