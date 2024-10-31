import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv
import sys
import json
import requests
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureMachineLearningSkill,
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIModelName,
    AzureOpenAIVectorizer,
    AzureOpenAIParameters,
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    FieldMapping,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    ScalarQuantizationCompressionConfiguration,
    ScalarQuantizationParameters,
    SimpleField,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjections,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SplitSkill,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import (
    HybridCountAndFacetMode,
    HybridSearch,
    SearchScoreThreshold,
    VectorSimilarityThreshold,
    VectorizableTextQuery,
    VectorizedQuery
)
import os
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery

# Load .env file
load_dotenv()

# Azure Search Service Information
service_name = os.getenv("AZURE_AI_SEARCH_ENDPOINT") 
index_name = "cogsrch-index-profile-vector"
api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")

# Create the Search Client
endpoint = f"{service_name}"
#credential = AzureKeyCredential(api_key)
credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY", "")) if len(os.getenv("AZURE_AI_SEARCH_API_KEY", "")) > 0 else DefaultAzureCredential()

# Function to retrieve current embeddings from the index
#def retrieve_embeddings_from_index(index_name):
#    # This will depend on the specific library you're using for the index
#    # For example, if using Pinecone, you might retrieve the vectors like this:
#    embeddings = your_index_library.fetch_all_embeddings(index_name)
#    return embeddings

# Function to convert text to new embeddings using 'text-embedding-ada-large'
def convert_text_to_new_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-large"
    )
    new_embedding = response['data'][0]['embedding']
    return new_embedding

# Function to update the new index with new embeddings
#def update_new_index(new_index_name, text_embedding_pairs):
#    # Assuming your index accepts text and new embeddings in pairs (text, embedding)
#    your_index_library.add_batch(new_index_name, text_embedding_pairs)

# Main function to handle the entire conversion process
def convert_embeddings(index_name, new_index_name):
    # Step 1: Retrieve all embeddings from the current index
    #old_embeddings = retrieve_embeddings_from_index(index_name)
    
    # Step 2: Convert each text from old embedding to the new embedding model
    new_text_embedding_pairs = []
    
    # for entry in old_embeddings:
    #     text = entry['text']  # Assuming each entry has associated text
    #     new_embedding = convert_text_to_new_embedding(text)
        
    #     # Create a tuple or dict with text and the new embedding
    #     new_text_embedding_pairs.append({
    #         'text': text,
    #         'embedding': new_embedding
    #     })
    
    # # Step 3: Insert the new embeddings into the new index
    # update_new_index(new_index_name, new_text_embedding_pairs)
    # print(f"Updated the new index '{new_index_name}' with new embeddings.")

def get_azureopenaiembedding(text):
    #response = openai.Embedding.create(
    #    input=text,
    #    model="text-embedding-ada-large"
    #)
    #new_embedding = response['data'][0]['embedding']

    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2024-06-01",
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )

    response = client.embeddings.create(
        input = text,
        model= "text-embedding-3-large"
    )

    print(response.model_dump_json(indent=2))
    new_embedding = response['data'][0]['embedding']

    return new_embedding

client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2024-06-01",
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT") 
    )

def generate_embeddings(text, model="text-embedding-3-large"): # model = "deployment_name"
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Search Index Schema definition
index_schema = "./newindex.json"
batch_size = 1000

# Instantiate a client
class CreateClient(object):
    def __init__(self, endpoint, key, index_name):
        self.endpoint = endpoint
        self.index_name = index_name
        self.key = key
        self.credentials = AzureKeyCredential(key)

    # Create a SearchClient
    # Use this to upload docs to the Index
    def create_search_client(self):
        return SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credentials,
        )

    # Create a SearchIndexClient
    # This is used to create, manage, and delete an index
    def create_admin_client(self):
        return SearchIndexClient(endpoint=endpoint, credential=self.credentials)


# Get Schema from File or URL
def get_schema_data(schema, url=False):
    if not url:
        with open(schema) as json_file:
            schema_data = json.load(json_file)
            return schema_data
    else:
        data_from_url = requests.get(schema)
        schema_data = json.loads(data_from_url.content)
        return schema_data


# Create Search Index from the schema
# If reading the schema from a URL, set url=True
def create_schema_from_json_and_upload(schema, index_name, admin_client, url=False):

    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []
    schema_data = get_schema_data(schema, url)

    index = SearchIndex(
        name=index_name,
        fields=schema_data["fields"],
        scoring_profiles=scoring_profiles,
        suggesters=schema_data["suggesters"],
        cors_options=cors_options,
    )

    try:
        upload_schema = admin_client.create_index(index)
        if upload_schema:
            print(f"Schema uploaded; Index created for {index_name}.")
        else:
            exit(0)
    except:
        print("Unexpected error:", sys.exc_info()[0])

# Batch your uploads to Azure Search
def batch_upload_json_data_to_index(json_file, client):
    batch_array = []
    count = 0
    batch_counter = 0
    for i in json_file:
        count += 1
        batch_array.append(
            {
                "id": str(i["id"]),
                "title": str(i["title"]),
                "chunk": str(i["chunk"]),
                "chunkVector": complex(i["chunkVector"]),
                "name": str(i["name"]),
                "location": str(i["location"]),
                "page_num": int(i["page_num"]),
            }
        )

        # In this sample, we limit batches to 1000 records.
        # When the counter hits a number divisible by 1000, the batch is sent.
        if count % batch_size == 0:
            client.upload_documents(documents=batch_array)
            batch_counter += 1
            print(f"Batch sent! - #{batch_counter}")
            batch_array = []

    # This will catch any records left over, when not divisible by 1000
    if len(batch_array) > 0:
        client.upload_documents(documents=batch_array)
        batch_counter += 1
        print(f"Final batch sent! - #{batch_counter}")

    print("Done!")

# https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/basic-vector-workflow/azure-search-vector-python-sample.ipynb
def createnewindex(new_index_name):
    # Create a search index
    index_client = SearchIndexClient(
        endpoint=endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchableField(name="name", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=True),
        SearchableField(name="chunk", type=SearchFieldDataType.String,
                        filterable=True),
        SearchField(name="chunkVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=3072, vector_search_profile_name="myHnswProfile"),
        SearchableField(name="location", type=SearchFieldDataType.String),
        SearchableField(name="page_num", type=SearchFieldDataType.Int32),
    ]

    # https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/e2e-demos/azure-ai-search-e2e-build-demo.ipynb
    # Configure the vector search configuration  
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer_name="myVectorizer"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name="myVectorizer",
                kind="azureOpenAI",
                azure_open_ai_parameters=AzureOpenAIParameters(
                    resource_uri=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    deployment_id="text-embedding-3-large",
                    model_name="text-embedding-3-large",
                    api_key=os.getenv("AZURE_OPENAI_API_KEY")
                )
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            keywords_fields=[SemanticField(field_name="name")],
            content_fields=[SemanticField(field_name="chunk")]
        )
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created')

# Replace with your index names
index_name = 'cogsrch-index-profile-vector'
new_index_name = 'cogsrch-index-profile-vector-large'
#https://github.com/Azure-Samples/azure-search-python-samples/blob/main/bulk-insert/bulk-insert.py

# Run the conversion
# convert_embeddings(index_name, new_index_name)

def searchvector():
    
    # Pure Vector Search
    query = "show me projects experience on railway construction"  
    embedding = client.embeddings.create(input=query, model="text-embedding-3-large", 
                                         dimensions=1536).data[0].embedding
    
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=5, fields="chunkVector")
    
    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["title", "name", "chunk"],
    )  
    
    for result in results:  
        print(f"Title: {result['title']}")  
        print(f"Score: {result['@search.score']}")  
        print(f"Chunk: {result['chunk']}")  
        print(f"Name: {result['name']}\n") 

def main():
    # Initialize the Search Client
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # Query all documents from the index
    results = search_client.search(search_text="*", top=10)  # You can customize query here, * gets all documents
    # https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/index-backup-restore/azure-search-backup-and-restore.ipynb

    # https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/basic-vector-workflow/azure-search-vector-python-sample.ipynb

    # searchvector()
    createnewindex(new_index_name)

    # Loop through each document and access the embeddings column
    for result in results:
        doc_id = result['@search.score']  # Unique identifier (you can use a different field if needed)
        if 'chunkVector' in result:  # Replace with the actual column name containing embeddings
            embeddings = result['chunkVector']
            chunk = result['chunk']
            id = result['id']
            title = result['title']
            name = result['name']
            location = result['location']
            page_num = result['page_num']
            # print(f"Document ID: {doc_id}, Embeddings: {embeddings}")
            #print(f"Document ID: {doc_id}, Chunk: {chunk}, Title: {title}, Name: {name}, Location: {location}, Page Number: {page_num} \n")
            embeddingsnew = generate_embeddings(chunk)
            #print("New embedding: ", embeddingsnew)
            doc = {
                "id": id,
                "title": title,
                "chunk": chunk,
                "chunkVector": embeddingsnew,
                "name": name,
                "page_num": page_num,
            }
            search_client = SearchClient(endpoint=endpoint, index_name=new_index_name, credential=credential)
            result = search_client.upload_documents(doc)
            print(f"Uploaded {len(doc)} documents") 
            print("*" * 100)
        else:
            print(f"Document ID: {doc_id} has no embeddings")

if __name__ == "__main__":
    main()