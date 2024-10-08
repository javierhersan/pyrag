from dotenv import load_dotenv
import os
from openai import OpenAI
import faiss
import numpy as np
import requests
import re
from pandas import DataFrame
import chromadb
from pathlib import Path

def main():
    # Load the .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        print(f"\n------------------------------------------------------------------------------")
        print("API_KEY is not set in the .env file")
        print(f"------------------------------------------------------------------------------\n")

    else:
        print(f"------------------------------------------------------------------------------")
        print(f"API Key: {api_key}")
        print(f"------------------------------------------------------------------------------\n")

    
    chromadb_path = Path(__file__).parent / "chromadb"
    chroma_client = chromadb.Client() # Transient Data for testing
    chroma_client = client = chromadb.PersistentClient(path= str(chromadb_path))

    client = OpenAI(api_key=api_key)

    print(f"------------------------------------------------------------------------------")
    print(f"Storing relevant documents...")
    documents = [
        """
            If the user wants to make a transaction (amount always in USD) answer with: <execute> transaction(address:str, usd_amount:int) </execute>
        """, 
        """This is a list of Zurich Business Units with its relevant bank accounts:
            Zurich Barcelona: ES00 0000 0000 0000
            Zurich Honk Kong: HK28 0000 0000 0000
            Zurich North America: NA99 0000 0000 0000
        """,
        """
            FX Rates:
            USD to EUR: 1.10
            AUD to EUR: 1.50
        """, 
    ]
    print(f"Relevant documents: \n")
    for doc in documents:
        print(f"{doc}")

    docs = DataFrame({"content":documents})
    docs["id"]=docs.index
    
    print(f"Relevant documents stored")
    print(f"------------------------------------------------------------------------------\n")

    print(f"------------------------------------------------------------------------------")
    print(f"Storing embeddings...")
    embeddings = get_embeddings(documents, client)
    
    collection_name = "chromadb_docs"
    if len(chroma_client.list_collections())>0 and collection_name in [chroma_client.list_collections()[0].name]:
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(name=collection_name)
    else:
        collection = chroma_client.create_collection(name=collection_name)

    dimension = embeddings.shape[1]

    # Create Vector Database Colection
    collection.add(
        documents=docs["content"].tolist(),
        metadatas=[{"ids": id} for id in docs["id"].tolist()],
        ids=[f"id{x}" for x in docs["id"].tolist()]
        # embeddings= ... # You can add open ai or other embeddings
    )

    print(embeddings)
    print(f"Embeddings stored")
    print(f"------------------------------------------------------------------------------\n")

    conversation_history = []
    while True:
        # query = "I want to send 50€ to ES00 0000 0000 0000 address"
        query = input("> User: ")
        conversation_history.append(f"> User: {query}")

        print(f"> Bot: Reading relevant documents...")
        relevant_documents = query_embeddings(query, collection)
        print(f"> Bot: Relevant documents found: {relevant_documents}")

        response = generate_response(conversation_history, client, relevant_documents)
        was_tool_executed = execute_tools(response)
        if (not was_tool_executed): print(f"> Bot: {response}")
        conversation_history.append(f"> Bot: {response}")

def get_embeddings(documents:list[str], client, model="text-embedding-ada-002"):
    #    text = text.replace("\n", " ")
    # embeddings =  client.embeddings.create(input = documents, model=model).data[0].embedding
    # embeddings = np.array([data['embedding'] for data in response['data']])
    response = client.embeddings.create(input = documents, model=model)
    embeddings = np.array([data.embedding for data in response.data])
    return embeddings

def query_embeddings(query:str, collection, k=3):
    documents = collection.query(
        query_texts=[query],
        # where={"topics":"SCIENCE"}, # You can add filtering
        n_results=k
    )
    return documents["documents"][0]
    # return [documents[i] for i in indices[0]]

def delete_doc(id:str, collection):
    collection.delete(ids=[id])

def get_doc(id:str, collection):
    collection.get(ids=[id])

def update_doc_metadata(id:str, metadata:str, collection):
    collection.update(ids=[id], metadatas=[{"meta":metadata}])

# Function to generate response using OpenAI's language model
def generate_response(conversation_history, client, relevant_documents):
    context = "\n".join(relevant_documents)
    conversation = "\n".join(conversation_history)
    prompt =    f"""    
                        You are an assistant that can answer questions or execute functions.
                        Relevant context information: {context}
                        Conversation history: {conversation}:
                """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def execute_tools(text:str):
    pattern = r'<execute>\s*(.*?)\s*</execute>'
    match = re.search(pattern, text)
    if match:
        code_to_execute = match.group(1).strip()
        code_to_execute = re.sub(r'\s+', ' ', code_to_execute)

        print(f"> Bot: Selecting tool: {code_to_execute}")
        print(f"> Bot: Executing tool...")
        # print(f"> Tool: 0%")
        # print(f"> Tool: 0% ↻ 25%")
        # print(f"> Tool: 0% ↻ 25% ↻ 50%")
        # print(f"> Tool: 0% ↻ 25% ↻ 50% ↻ 75%")
        # print(f"> Tool: 0% ↻ 25% ↻ 50% ↻ 75% ↻ 100% ")
        
        # Define the transaction function
        def transaction(address, usd_amount):
            print(f"> Tool: Transaction executed | address={address}, amount={usd_amount}")

        # Execute the extracted code
        exec(code_to_execute)
        print("> Bot: Tool executed")
        return True
    else:
        return False

if __name__ == "__main__":
    main()


