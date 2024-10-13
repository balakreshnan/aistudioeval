import pandas as pd
import os

from pprint import pprint
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import RelevanceEvaluator
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from askwiki import ask_wiki

ask_wiki(query="What is the capital of India?")

def main():
    df = pd.read_json("data.jsonl", lines=True)
    print(df.head())
    model_config = {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_KEY"),
        "azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    }

    azure_ai_project={
        "subscription_id": os.environ.get("subscription_id"),
        "resource_group_name": os.environ.get("resource_group_name"),
        "project_name": os.environ.get("babal-sweden"),
    }

    relevance_evaluator = RelevanceEvaluator(model_config)

    relevance_evaluator(
        response="New Delhi is Capital of India",
        context="India is a country in South Asia.",
        query="What is the capital of India?",
    )

    results = evaluate(
        data="data.jsonl",
        target=ask_wiki,
        evaluators={
            "relevance": relevance_evaluator,
        },
        azure_ai_project=azure_ai_project,
    )
    pprint(results)
    pd.DataFrame(results["rows"])

if __name__ == "__main__":
    main()