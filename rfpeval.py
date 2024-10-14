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
    
    citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    # print(citationtxt)
    model_config = {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    }
    
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)

    azure_ai_project={
        "subscription_id": os.environ.get("subscription_id"),
        "resource_group_name": os.environ.get("resource_group_name"),
        "project_name": os.environ.get("project_name"),
        "azure_crendential": credential,
    }

    relevance_evaluator = RelevanceEvaluator(model_config)

    relevance_evaluator(
        response=citationtxt,
        context="summary of Resources for Railway projects.",
        query="Provide summary of Resources for Railway projects with 200 words?",
    )
    # pprint(relevance_evaluator)

    results = evaluate(
        evaluation_name="rfp_evaluation",
        data="datarfp.jsonl",
        target=extractrfpresults,
        evaluators={
            "relevance": relevance_evaluator,
        },
        evaluator_config={
            "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        },
        # azure_ai_project=azure_ai_project,
        output_path="./rsoutputmetrics.json",
    )
    # pprint(results)
    parse_json(results)

if __name__ == "__main__":
    main()