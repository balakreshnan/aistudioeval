import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
from datetime import timedelta
import json
from PIL import Image
import base64
import requests
import io
from typing import Optional
from typing_extensions import Annotated
import wave
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
from flows.rfpdata import processpdfwithprompt, extractrfpresults

# Load .env file
load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-05-01-preview",
)

model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def main():
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway construction projects with 200 words?")
    citationtxt = extractrfpresults("Provide summary of Resources for Construction projects with 200 words?")

    print('Citation Text:', citationtxt)

if __name__ == "__main__":
    main()