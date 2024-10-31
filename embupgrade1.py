import requests
import json

import json
import numpy as np
import os
import pandas as pd
import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from dotenv import load_dotenv

# Load .env file
load_dotenv()

#https://raw.githubusercontent.com/memasanz/CogSearchOpenAIWithOutVectorSearch/refs/heads/main/cog_search.py

# Azure Search Service Information
service_name = os.getenv("AZURE_AI_SEARCH_ENDPOINT_NAME") 
index_name = "cogsrch-index-profile-vector"
api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
new_index_name = "cogsrch-index-profile-vector-large"

# Create the Search Client
#endpoint = f"{service_name}"
endpoint = "https://{}.search.windows.net/".format(service_name)
#credential = AzureKeyCredential(api_key)
credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY", "")) if len(os.getenv("AZURE_AI_SEARCH_API_KEY", "")) > 0 else DefaultAzureCredential()

# Function to get embedding
def get_embedding(text, model="text-embedding-ada-002"):
    # In newer versions, the embedding endpoint has been updated
    embedding = openai.embeddings.create(input=text, model=model)['data'][0]['embedding']
    return embedding

def search(question):
    response = openai.embeddings.create(input=question,model="text-embedding-ada-002")
    #print(response)
    #q_embeddings = response['data'][0]['embedding'][0]['embedding=']
    q_embeddings = response.data[0].embedding
    
    if len(question) > 0:
        endpoint = "https://{}.search.windows.net/".format(service_name)
        url = '{0}indexes/{1}/docs/search?api-version=2024-07-01'.format(endpoint, index_name)

        print(url)

        payload = json.dumps({
        "search": question,
        "count": True,
        })
        headers = {
        'api-key': '{0}'.format(api_key),
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        obj = response.json()
        relevant_data = []
        lst_embeddings_text = []
        lst_embeddings = []
        lst_file_name = []
        count = 0
        for x in obj['value']:
            if x['@search.score'] > 0.5:
                count += 1
                relevant_data.append(x['chunk'])
                embeddings = x['chunkVector']
                embeddings_text = x['chunk']
                file_name = x['title']

                # curie_search = []
                # for x in embeddings:
                #     a = np.fromstring(x[1:-1], dtype=float, sep=',')
                #     curie_search.append(a)
                # curie_list = list(curie_search)

                # for i in range(len(embeddings)):
                #     lst_embeddings_text.append(embeddings_text[i])
                #     lst_embeddings.append(np.fromstring(embeddings[i][1:-1], dtype=float, sep=','))
                #     lst_file_name.append(file_name)
            
            print(x['chunk'])


        return relevant_data, count, lst_file_name

def create_index_semantic():

    #url = '{0}/indexes/{1}/?api-version=2021-04-30-Preview'.format(self.endpoint, self.index)
    endpoint = "https://{}.search.windows.net/".format(service_name)
    url = '{0}/indexes/{1}/?api-version=2024-07-01'.format(endpoint, new_index_name)    
    print(url)

    payload = json.dumps({
    "name": new_index_name,
    "defaultScoringProfile": "",
    "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "searchable": True,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": True,
      "key": True,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "title",
      "type": "Edm.String",
      "searchable": True,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": True,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "chunk",
      "type": "Edm.String",
      "searchable": True,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": True,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "chunkVector",
      "type": "Collection(Edm.Single)",
      "searchable": True,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": 1536,
      "vectorSearchProfile": "vector-profile-1",
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "name",
      "type": "Edm.String",
      "searchable": True,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "location",
      "type": "Edm.String",
      "searchable": False,
      "filterable": False,
      "retrievable": True,
      "stored": True,
      "sortable": False,
      "facetable": False,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    },
    {
      "name": "page_num",
      "type": "Edm.Int32",
      "searchable": False,
      "filterable": True,
      "retrievable": True,
      "stored": True,
      "sortable": True,
      "facetable": True,
      "key": False,
      "indexAnalyzer": None,
      "searchAnalyzer": None,
      "analyzer": None,
      "dimensions": None,
      "vectorSearchProfile": None,
      "vectorEncoding": None,
      "synonymMaps": []
    }
  ],
  "scoringProfiles": [],
  "corsOptions": None,
  "suggesters": [],
  "analyzers": [],
  "tokenizers": [],
  "tokenFilters": [],
  "charFilters": [],
  "encryptionKey": None,
  "similarity": {
    "@odata.type": "#Microsoft.Azure.Search.BM25Similarity",
    "k1": None,
    "b": None
  },
  "semantic": {
    "defaultConfiguration": None,
    "configurations": [
      {
        "name": "my-semantic-config",
        "prioritizedFields": {
          "titleField": {
            "fieldName": "title"
          },
          "prioritizedContentFields": [
            {
              "fieldName": "chunk"
            }
          ],
          "prioritizedKeywordsFields": []
        }
      }
    ]
  },
  "vectorSearch": {
    "algorithms": [
      {
        "name": "vectorConfig",
        "kind": "hnsw",
        "hnswParameters": {
          "metric": "cosine",
          "m": 4,
          "efConstruction": 400,
          "efSearch": 500
        },
        "exhaustiveKnnParameters": None
      }
    ],
    "profiles": [{
                "name": "vector-profile-1",
                "algorithm": "vectorConfig"
            }],
    "vectorizers": [],
    "compressions": []
  }
    })
    headers = {
    'api-key': api_key,
    'Content-Type': 'application/json'
    }

    response = requests.request("PUT", url, headers=headers, data=payload)
    print(response.text)

    if response.status_code == 201 or response.status_code == 204:
        print('good')
        return response, True
    else:
        # print('************************')
        # print(response.status_code)
        # print(response.text)
        return response, False
    
def create_indexer(self):

    url = '{0}/indexers/{1}-indexer/?api-version=2024-07-01'.format(endpoint, new_index_name)
    print(url)

    payload = json.dumps({
    "name": "{0}-indexer".format(new_index_name),
    "description": "",
    "dataSourceName": "{0}-datasource".format(new_index_name),
    "skillsetName": "{0}-skillset".format(new_index_name),
    "targetIndexName": "{0}".format(new_index_name),
    "disabled": None,
    "schedule": None,
    "parameters": {
        "batchSize": None,
        "maxFailedItems": 0,
        "maxFailedItemsPerBatch": 0,
        "base64EncodeKeys": None,
        "configuration": {
        "dataToExtract": "contentAndMetadata",
        "parsingMode": "default",
        "imageAction": "generateNormalizedImages"
        }
    },
    "fieldMappings": [
        {
        "sourceFieldName": "metadata_storage_path",
        "targetFieldName": "metadata_storage_path",
        "mappingFunction": {
            "name": "base64Encode",
            "parameters": None
        }
        }
    ],
    "cache": None,
    "encryptionKey": None
    })
    headers = {
    'Content-Type': 'application/json',
    'api-key': '{0}'.format(self.search_key)
    }


    response = requests.request("PUT", url, headers=headers, data=payload)


    if response.status_code == 201 or response.status_code == 204:
        print('good')
        return response, True
    else:
        print(response.status_code)
        return response, False
    
def main():

    #search("show me candidates for technical strategy roles")
    #create_index_semantic()
    # search old index and write back to new index


if __name__ == "__main__":
    main()