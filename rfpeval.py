import pandas as pd
import os

from pprint import pprint
from azure.ai.evaluation import evaluate
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import RelevanceEvaluator
from azure.ai.evaluation import (
    ContentSafetyEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    GroundednessEvaluator,
    FluencyEvaluator,
    SimilarityEvaluator,
    ViolenceEvaluator,
    SexualEvaluator,
    SelfHarmEvaluator,
    HateUnfairnessEvaluator,
)
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from rfpdata import processpdfwithprompt, extractrfpresults

# Function to parse the JSON data
def parse_json(data):
    # Print Relevance Score
    print(f"Overall GPT Relevance: {data.get('relevance.gpt_relevance', 'N/A')}")
    
    # Print Rows
    rows = data.get('rows', [])
    print("\nRows:")
    for row in rows:
        context = row.get('inputs.context')
        query = row.get('inputs.query')
        response = row.get('inputs.response')
        output = row.get('outputs.output')
        relevance = row.get('outputs.relevance.gpt_relevance')
        
        print(f"Context: {context}")
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Output: {output}")
        print(f"Relevance: {relevance}")
        print("-" * 50)

def main():
    
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    # print(citationtxt)
    model_config = {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION"),
    }

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)

    azure_ai_project={
        "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
        "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
        "project_name": os.environ.get("AZUREAI_PROJECT_NAME"),
        # "azure_crendential": credential,
    }

    #relevance_evaluator = RelevanceEvaluator(model_config)

    #relevance_evaluator(
    #    response=citationtxt,
    #    context="summary of Resources for Railway projects.",
    #    query="Provide summary of Resources for Railway projects with 200 words?",
    #)
    # pprint(relevance_evaluator)

    # prompty_path = os.path.join("./", "rfp.prompty")
    content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project)
    relevance_evaluator = RelevanceEvaluator(model_config)
    coherence_evaluator = CoherenceEvaluator(model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    similarity_evaluator = SimilarityEvaluator(model_config)

    results = evaluate(
        evaluation_name="rfpevaluation",
        data="datarfp.jsonl",
        target=extractrfpresults,
        evaluators={
            "relevance": relevance_evaluator,
        },
        evaluator_config={
            "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        },
        #evaluators={
        #    "content_safety": content_safety_evaluator,
        #    "coherence": coherence_evaluator,
        #    "relevance": relevance_evaluator,
        #    "groundedness": groundedness_evaluator,
        #    "fluency": fluency_evaluator,
        #    "similarity": similarity_evaluator,
        #},        
        #evaluator_config={
        #    "content_safety": {"query": "${data.query}", "response": "${target.response}"},
        #    "coherence": {"response": "${target.response}", "query": "${data.query}"},
        #    "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        #    "groundedness": {
        #        "response": "${target.response}",
        #        "context": "${data.context}",
        #        "query": "${data.query}",
        #    },
        #    "fluency": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        #    "similarity": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        #},
        # azure_ai_project=azure_ai_project,
        output_path="./rsoutputmetrics.json",
    )
    # pprint(results)
    parse_json(results)

if __name__ == "__main__":    
    main()