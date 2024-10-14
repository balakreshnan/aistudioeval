from azure.ai.evaluation import BleuScoreEvaluator
from azure.ai.evaluation import GleuScoreEvaluator
from azure.ai.evaluation import MeteorScoreEvaluator
from azure.ai.evaluation import RougeScoreEvaluator, RougeType
from azure.ai.evaluation import evaluate
import os
from dotenv import load_dotenv
from pprint import pprint

# Load .env file
load_dotenv()



def main():

    azure_ai_project={
        "subscription_id": os.environ.get("subscription_id"),
        "resource_group_name": os.environ.get("resource_group_name"),
        "project_name": os.environ.get("project_name"),
    }
    bleu = BleuScoreEvaluator()
    gleu = GleuScoreEvaluator()
    meteor = MeteorScoreEvaluator()
    rouge = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_L)

    result = gleu(response="Tokyo is the capital of Japan.", ground_truth="The capital of Japan is Tokyo.")

    print(result)

    result = evaluate(
    data="datamath.jsonl",
    evaluators={
        #"bleu": bleu,
        "gleu": gleu,
        "meteor": meteor,
        "rouge": rouge,
    },
    # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
    azure_ai_project=azure_ai_project,
)
    pprint(result)

if __name__ == "__main__":
    main()