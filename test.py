import pandas as pd
import os

from pprint import pprint
from azure.ai.evaluation import evaluate

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from flows.rfpdata import processpdfwithprompt, extractrfpresults

def main():
    print('Processing RFP with prompt...')
    citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    print('Citation Text:', citationtxt)

if __name__ == "__main__":
    main()