import asyncio
from pathlib import Path
import pandas as pd
import os

from pprint import pprint
from azure.ai.evaluation import evaluate, AzureAIProject, AzureOpenAIModelConfiguration
from azure.ai.evaluation import ProtectedMaterialEvaluator, IndirectAttackEvaluator
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario, IndirectAttackSimulator
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
from typing import Any, List, Dict, Optional
from openai import AzureOpenAI

# Load .env file
load_dotenv()

from flows.rfpdata import processpdfwithprompt, extractrfpresults

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

async def protected_material_callback(
    messages: List[Dict], stream: bool = False, session_state: Optional[str] = None, context: Optional[Dict] = None
) -> dict:
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # Get a client handle for the model
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-05-01-preview",
    )
    # Call the model
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": messages["messages"][0]["content"],  # injection of prompt happens here.
            }
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )

    formatted_response = completion.to_dict()["choices"][0]["message"]
    messages["messages"].append(formatted_response)
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context,
    }


async def test_protected_material():
    # Load .env file
    # Load .env file
    load_dotenv()
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    model_config = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "azure_deployment": azure_deployment,
        "api_version": api_version,
    }


    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)


    # azure_ai_project={
    #     "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
    #     "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP"),
    #     "project_name": os.getenv("AZUREAI_PROJECT_NAME"),
    #     # "azure_crendential": credential,
    # }
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZUREAI_PROJECT_NAME")
    print(subscription_id, resource_group_name, project_name)
    azure_ai_project = AzureAIProject(subscription_id=subscription_id, 
                                      resource_group_name=resource_group_name, 
                                      project_name=project_name, 
                                      azure_crendential=credential)
    
    azure_ai_project_dict = {
        "subscription_id": subscription_id,
        "resource_group_name": resource_group_name,
        "project_name": project_name,
        "azure_credential": credential
    }
    print("Protected Materials Evaluation")
    # initialize the adversarial simulator
    protected_material_simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)

    # define the adversarial scenario you want to simulate
    protected_material_scenario = AdversarialScenario.ADVERSARIAL_CONTENT_PROTECTED_MATERIAL
    unfiltered_protected_material_outputs = await protected_material_simulator(
        scenario=protected_material_scenario,
        max_conversation_turns=3,  # define the number of conversation turns
        max_simulation_results=10,  # define the number of simulation results
        target=protected_material_callback,  # define the target model callback
    )
    # Results are truncated for brevity.
    truncation_limit = 50
    for output in unfiltered_protected_material_outputs:
        for turn in output["messages"]:
            print(f"{turn['role']} : {turn['content'][0:truncation_limit]}")
    from pathlib import Path

    print(unfiltered_protected_material_outputs.to_eval_qr_json_lines())
    output = unfiltered_protected_material_outputs.to_eval_qr_json_lines()
    file_path = "unfiltered_protected_material_output.jsonl"

    # Write the output to the file
    with Path.open(Path(file_path), "w") as file:
        file.write(output)
    
    protected_material_eval = ProtectedMaterialEvaluator(azure_ai_project=azure_ai_project, credential=credential)

    result = evaluate(
        data=file_path,
        evaluators={"protected_material": protected_material_eval},
        # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
        azure_ai_project=azure_ai_project,
        # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
        output_path="./mynewfilteredIPevalresults.json",
    )
    filtered_protected_material_outputs = await protected_material_simulator(
        scenario=protected_material_scenario,
        max_conversation_turns=3,  # define the number of conversation turns
        max_simulation_results=10,  # define the number of simulation results
        target=protected_material_callback,  # now with the Prompt Shield attached to our model deployment
    )
    print(filtered_protected_material_outputs.to_eval_qr_json_lines())
    output = filtered_protected_material_outputs.to_eval_qr_json_lines()
    filtered_protected_material_file_path = "filtered_protected_material_output.jsonl"

    # Write the output to the file
    with Path.open(Path(filtered_protected_material_file_path), "w") as file:
        file.write(output)
    
    filtered_result = evaluate(
        data=filtered_protected_material_file_path,
        evaluators={"protected_material": protected_material_eval},
        # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
        azure_ai_project=azure_ai_project,
        # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
        output_path="./myfilteredevalresults.json",
    )

async def xpia_callback(
    messages: List[Dict], stream: bool = False, session_state: Optional[str] = None, context: Optional[Dict] = None
) -> dict:
    messages_list = messages["messages"]
    # get last message
    latest_message = messages_list[-1]
    query = latest_message["content"]
    context = None
    if "file_content" in messages["template_parameters"]:
        query += messages["template_parameters"]["file_content"]
    # the next few lines explain how to use the AsyncAzureOpenAI's chat.completions
    # to respond to the simulator. You should replace it with a call to your model/endpoint/application
    # make sure you pass the `query` and format the response as we have shown below

    # Get a client handle for the model
    deployment = os.environ.get("AZURE_DEPLOYMENT_NAME")

    #token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    oai_client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-05-01-preview",
    )
    try:
        response_from_oai_chat_completions = oai_client.chat.completions.create(
            messages=[{"content": query, "role": "user"}], model=deployment, max_tokens=300
        )
        print(response_from_oai_chat_completions)
    except Exception as e:
        print(f"Error: {e} with content length {len(query)}")
        # to continue the conversation, return the messages, else you can fail the adversarial with an exception
        message = {
            "content": "Something went wrong. Check the exception e for more details.",
            "role": "assistant",
            "context": None,
        }
        messages["messages"].append(message)
        return {"messages": messages["messages"], "stream": stream, "session_state": session_state}
    response_result = response_from_oai_chat_completions.choices[0].message.content
    formatted_response = {
        "content": response_result,
        "role": "assistant",
        "context": {},
    }
    messages["messages"].append(formatted_response)
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}

async def jailbreak():
    load_dotenv()
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    model_config = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "azure_deployment": azure_deployment,
        "api_version": api_version,
    }


    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZUREAI_PROJECT_NAME")
    print(subscription_id, resource_group_name, project_name)
    azure_ai_project = AzureAIProject(subscription_id=subscription_id, 
                                      resource_group_name=resource_group_name, 
                                      project_name=project_name, 
                                      azure_crendential=credential)
    
    azure_ai_project_dict = {
        "subscription_id": subscription_id,
        "resource_group_name": resource_group_name,
        "project_name": project_name,
        "azure_credential": credential
    }
    indirect_attack_simulator = IndirectAttackSimulator(
        azure_ai_project=azure_ai_project, credential=DefaultAzureCredential()
    )

    unfiltered_indirect_attack_outputs = await indirect_attack_simulator(
        target=xpia_callback,
        scenario=AdversarialScenario.ADVERSARIAL_INDIRECT_JAILBREAK,
        max_simulation_results=10,
        max_conversation_turns=3,
    )
    pprint(unfiltered_indirect_attack_outputs)
    # Results are truncated for brevity.
    truncation_limit = 50
    for output in unfiltered_indirect_attack_outputs:
        for turn in output["messages"]:
            content = turn["content"]
            if isinstance(content, dict):  # user response from callback is dict
                print(f"{turn['role']} : {content['content'][0:truncation_limit]}")
            elif isinstance(content, tuple):  # assistant response from callback is tuple
                print(f"{turn['role']} : {content[0:truncation_limit]}")

    print(unfiltered_indirect_attack_outputs)
    print(unfiltered_indirect_attack_outputs.to_eval_qr_json_lines())
    output = unfiltered_indirect_attack_outputs.to_eval_qr_json_lines()
    xpia_file_path = "unfiltered_indirect_attack_outputs.jsonl"

    # Write the output to the file
    with Path.open(Path(xpia_file_path), "w") as file:
        file.write(output)
    
    indirect_attack_eval = IndirectAttackEvaluator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential())
    file_path = "indirect_attack_outputs.jsonl"
    result = evaluate(
        data=xpia_file_path,
        evaluators={
            "indirect_attack": indirect_attack_eval,
        },
        # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
        azure_ai_project=azure_ai_project,
        # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
        output_path="./mynewindirectattackevalresults.json",
    )

    filtered_indirect_attack_outputs = await indirect_attack_simulator(
        target=xpia_callback,  # now with the Prompt Shield attached to our model deployment
        scenario=AdversarialScenario.ADVERSARIAL_INDIRECT_JAILBREAK,
        max_simulation_results=10,
        max_conversation_turns=3,
    )
    print(filtered_indirect_attack_outputs)
    print(filtered_indirect_attack_outputs.to_eval_qr_json_lines())
    output = filtered_indirect_attack_outputs.to_eval_qr_json_lines()
    xpia_file_path = "filtered_indirect_attack_outputs.jsonl"

    # Write the output to the file
    with Path.open(Path(xpia_file_path), "w") as file:
        file.write(output)
    
    filtered_indirect_attack_result = evaluate(
        data=xpia_file_path,
        evaluators={"indirect_attack": indirect_attack_eval},
        # Optionally provide your AI Studio project information to track your evaluation results in your Azure AI Studio project
        azure_ai_project=azure_ai_project,
        # Optionally provide an output path to dump a json of metric summary, row level data and metric and studio URL
        output_path="./myindirectattackevalresults.json",
    )

def call_endpoint(query: str) -> dict:
    # token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    # Get a client handle for the model
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-05-01-preview",
    )
    # Call the model
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ],
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
    )
    return completion.to_dict()

async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,  # noqa: ANN401
    context: Optional[Dict[str, Any]] = None,
) -> dict:
    messages_list = messages["messages"]
    query = messages_list[-1]["content"]
    context = None
    try:
        response = call_endpoint(query)
        # we are formatting the response to follow the openAI chat protocol format
        formatted_response = {
            "content": response["choices"][0]["message"]["content"],
            "role": "assistant",
            "context": {context},
        }
    except Exception as e:
        response = f"Something went wrong {e!s}"
        formatted_response = None
    messages["messages"].append(formatted_response)
    return {"messages": messages_list, "stream": stream, "session_state": session_state, "context": context}

async def adversial_simulation():
    # Load .env file
    load_dotenv()
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    model_config = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "azure_deployment": azure_deployment,
        "api_version": api_version,
    }


    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)

    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZUREAI_PROJECT_NAME")
    print(subscription_id, resource_group_name, project_name)
    azure_ai_project = AzureAIProject(subscription_id=subscription_id, 
                                      resource_group_name=resource_group_name, 
                                      project_name=project_name, 
                                      azure_crendential=credential)
    
    azure_ai_project_dict = {
        "subscription_id": subscription_id,
        "resource_group_name": resource_group_name,
        "project_name": project_name,
        "azure_credential": credential
    }
    simulator = AdversarialSimulator(azure_ai_project=azure_ai_project, credential=credential)
    outputs = await simulator(
        scenario=AdversarialScenario.ADVERSARIAL_QA, max_conversation_turns=1, max_simulation_results=1, target=callback
    )
    #try:
    #    with Path.open("outputs1.jsonl", "w") as f:
    #        f.write(outputs.to_eval_qr_json_lines())
    #except Exception as e:
    #    print(e)

def evalmetrics():
    
    # Load .env file
    load_dotenv()
    #citationtxt = extractrfpresults("Provide summary of Resources for Railway projects with 200 words?")

    #print('Citation Text:', citationtxt)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    model_config = {
        "azure_endpoint": azure_endpoint,
        "api_key": api_key,
        "azure_deployment": azure_deployment,
        "api_version": api_version,
    }


    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        print(ex)


    # azure_ai_project={
    #     "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
    #     "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP"),
    #     "project_name": os.getenv("AZUREAI_PROJECT_NAME"),
    #     # "azure_crendential": credential,
    # }
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
    project_name = os.getenv("AZUREAI_PROJECT_NAME")
    print(subscription_id, resource_group_name, project_name)
    azure_ai_project = AzureAIProject(subscription_id=subscription_id, 
                                      resource_group_name=resource_group_name, 
                                      project_name=project_name, 
                                      azure_crendential=credential)
    
    azure_ai_project_dict = {
        "subscription_id": subscription_id,
        "resource_group_name": resource_group_name,
        "project_name": project_name,
        "azure_credential": credential
    }
    

    # relevance_evaluator = RelevanceEvaluator(model_config)

    # relevance_evaluator(
    #     response="Virginia railway express RFP need introduction, resources.",
    #     context="summary of Resources for Railway projects.",
    #     query="Provide summary of Resources for Railway projects with 200 words?",
    # )
    # pprint(relevance_evaluator)

    # prompty_path = os.path.join("./", "rfp.prompty")
    content_safety_evaluator = ContentSafetyEvaluator(azure_ai_project=azure_ai_project_dict, credential=credential)
    relevance_evaluator = RelevanceEvaluator(model_config)
    coherence_evaluator = CoherenceEvaluator(model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config)
    fluency_evaluator = FluencyEvaluator(model_config)
    # similarity_evaluator = SimilarityEvaluator(model_config)

    results = evaluate(
        evaluation_name="rfpevaluation",
        data="datarfp.jsonl",
        target=extractrfpresults,
        #evaluators={
        #    "relevance": relevance_evaluator,
        #},
        #evaluator_config={
        #    "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        #},
        evaluators={
            "content_safety": content_safety_evaluator,
            "coherence": coherence_evaluator,
            "relevance": relevance_evaluator,
            "groundedness": groundedness_evaluator,
            "fluency": fluency_evaluator,
        #    "similarity": similarity_evaluator,
        },        
        evaluator_config={
            "content_safety": {"query": "${data.query}", "response": "${target.response}"},
            "coherence": {"response": "${target.response}", "query": "${data.query}"},
            "relevance": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
            "groundedness": {
                "response": "${target.response}",
                "context": "${data.context}",
                "query": "${data.query}",
            },
            "fluency": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
            "similarity": {"response": "${target.response}", "context": "${data.context}", "query": "${data.query}"},
        },
        azure_ai_project=azure_ai_project,
        output_path="./rsoutputmetrics.json",
    )
    # pprint(results)
    # parse_json(results)
    print("Done")

def main():
    #evalmetrics()
    #asyncio.run(test_protected_material())
    asyncio.run(jailbreak())
    #asyncio.run(adversial_simulation())

if __name__ == "__main__":    
    main()