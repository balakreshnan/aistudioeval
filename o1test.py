import pandas as pd
import os

from pprint import pprint
from azure.ai.evaluation import evaluate
from openai import AzureOpenAI

from dotenv import load_dotenv

# Load .env file
load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_O1_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_O1_KEY"),  
  api_version="2024-08-01-preview",
)

def processo1text(query):
    returntxt = ""

    rfttext = ""

    message_text = [
    {"role":"user", "content":f"""I am providing 3 datasets one for meta data, one for alarms and event, one for historical data with sensor. Your job is to analyze the latest fault and find the root cause from which equipment and subsystem it originated.

    Here is the meta data:
    Machine_ID,Machine_Name,Location,Sensors_Associated  
    M001,Conveyor Belt,Section A,"Sensor_1, Sensor_2"  
    M002,Heat Exchanger,Section B,"Sensor_3, Sensor_4"  
    M003,Compressor,Section C,"Sensor_5, Sensor_6"  
    M004,Hydraulic Press,Section D,"Sensor_7, Sensor_8"  
    M005,Packaging Machine,Section E,"Sensor_9, Sensor_10"  
    M006,Assembly Robot,Section F,"Sensor_11, Sensor_12"  
    M007,Paint Booth,Section G,"Sensor_13, Sensor_14"  
    M008,CNC Machine,Section H,"Sensor_15, Sensor_16"  
    M009,Lathe Machine,Section I,"Sensor_17, Sensor_18"  
    M010,Welding Station,Section J,"Sensor_19, Sensor_20"  
    M011,Drill Press,Section K,"Sensor_21, Sensor_22"  
    M012,Injection Molder,Section L,"Sensor_23, Sensor_24"  
    M013,Grinder,Section M,"Sensor_25, Sensor_26"  
    M014,Shearing Machine,Section N,"Sensor_27, Sensor_28"  
    M015,Water Jet Cutter,Section O,"Sensor_29, Sensor_30"  
    M016,Laser Cutter,Section P,"Sensor_31, Sensor_32"  
    M017,3D Printer,Section Q,"Sensor_33, Sensor_34"  
    M018,Printing Press,Section R,"Sensor_35, Sensor_36"  
    M019,Textile Loom,Section S,"Sensor_37, Sensor_38"  
    M020,Blender,Section T,"Sensor_39, Sensor_40"  
    M021,Extruder,Section U,"Sensor_41, Sensor_42"  
    M022,Granulator,Section V,"Sensor_43, Sensor_44"  
    M023,Boiler,Section W,"Sensor_45, Sensor_46"  
    M024,Reactor,Section X,"Sensor_47, Sensor_48"  
    M025,Filter,Section Y,"Sensor_49, Sensor_50"  


    Here is the alarms and event data:

    Timestamp,Event_ID,Event_Description,Severity,Related_Sensor  
    2023-01-01 02:00,E001,Overheat Detected,High,Sensor_3  
    2023-01-01 04:00,E002,Low Pressure Alert,Medium,Sensor_5  
    2023-01-01 06:00,E003,Vibration Alert,High,Sensor_7  
    2023-01-01 08:00,E004,High Temperature,Critical,Sensor_9  
    2023-01-01 10:00,E005,Power Surge Detected,High,Sensor_10  
    2023-01-01 12:00,E006,Overheat Detected,High,Sensor_3  
    2023-01-01 14:00,E007,Low Pressure Alert,Medium,Sensor_5  
    2023-01-01 16:00,E008,Vibration Alert,High,Sensor_7  
    2023-01-01 18:00,E009,High Temperature,Critical,Sensor_9  
    2023-01-01 20:00,E010,Power Surge Detected,High,Sensor_10  
    2023-01-01 22:00,E011,Overheat Detected,High,Sensor_3  
    2023-01-02 00:00,E012,Low Pressure Alert,Medium,Sensor_5  
    2023-01-02 02:00,E013,Vibration Alert,High,Sensor_7  
    2023-01-02 04:00,E014,High Temperature,Critical,Sensor_9  
    2023-01-02 06:00,E015,Power Surge Detected,High,Sensor_10  
    2023-01-02 08:00,E016,Overheat Detected,High,Sensor_3  
    2023-01-02 10:00,E017,Low Pressure Alert,Medium,Sensor_5  
    2023-01-02 12:00,E018,Vibration Alert,High,Sensor_7  
    2023-01-02 14:00,E019,High Temperature,Critical,Sensor_9  
    2023-01-02 16:00,E020,Power Surge Detected,High,Sensor_10  
    2023-01-02 18:00,E021,Overheat Detected,High,Sensor_3  
    2023-01-02 20:00,E022,Low Pressure Alert,Medium,Sensor_5  
    2023-01-02 22:00,E023,Vibration Alert,High,Sensor_7  
    2023-01-03 00:00,E024,High Temperature,Critical,Sensor_9  
    2023-01-03 02:00,E025,Power Surge Detected,High,Sensor_10  
    2023-01-03 04:00,E026,Overheat Detected,High,Sensor_3  
    2023-01-03 06:00,E027,Low Pressure Alert,Medium,Sensor_5  
    2023-01-03 08:00,E028,Vibration Alert,High,Sensor_7  
    2023-01-03 10:00,E029,High Temperature,Critical,Sensor_9  
    2023-01-03 12:00,E030,Power Surge Detected,High,Sensor_10  
    2023-01-03 14:00,E031,Overheat Detected,High,Sensor_3  
    2023-01-03 16:00,E032,Low Pressure Alert,Medium,Sensor_5  
    2023-01-03 18:00,E033,Vibration Alert,High,Sensor_7  
    2023-01-03 20:00,E034,High Temperature,Critical,Sensor_9  
    2023-01-03 22:00,E035,Power Surge Detected,High,Sensor_10  
    2023-01-04 00:00,E036,Overheat Detected,High,Sensor_3  
    2023-01-04 02:00,E037,Low Pressure Alert,Medium,Sensor_5  
    2023-01-04 04:00,E038,Vibration Alert,High,Sensor_7  
    2023-01-04 06:00,E039,High Temperature,Critical,Sensor_9  
    2023-01-04 08:00,E040,Power Surge Detected,High,Sensor_10  
    2023-01-04 10:00,E041,Overheat Detected,High,Sensor_3  
    2023-01-04 12:00,E042,Low Pressure Alert,Medium,Sensor_5  
    2023-01-04 14:00,E043,Vibration Alert,High,Sensor_7  
    2023-01-04 16:00,E044,High Temperature,Critical,Sensor_9  
    2023-01-04 18:00,E045,Power Surge Detected,High,Sensor_10  
    2023-01-04 20:00,E046,Overheat Detected,High,Sensor_3  
    2023-01-04 22:00,E047,Low Pressure Alert,Medium,Sensor_5  
    2023-01-05 00:00,E048,Vibration Alert,High,Sensor_7  
    2023-01-05 02:00,E049,High Temperature,Critical,Sensor_9  
    2023-01-05 04:00,E050,Power Surge Detected,High,Sensor_10  


    Here is the historical data:
    Timestamp,Sensor_1,Sensor_2,Sensor_3,Sensor_4,Sensor_5,Sensor_6,Sensor_7,Sensor_8,Sensor_9,Sensor_10  
    2023-01-01 00:00,0.50,1.20,0.90,0.70,1.10,0.80,0.60,1.00,0.40,0.80  
    2023-01-01 01:00,0.55,1.15,0.95,0.75,1.15,0.85,0.65,1.05,0.45,0.85  
    2023-01-01 02:00,0.60,1.10,1.00,0.80,1.20,0.90,0.70,1.10,0.50,0.90  
    2023-01-01 03:00,0.65,1.05,1.05,0.85,1.25,0.95,0.75,1.15,0.55,0.95  
    2023-01-01 04:00,0.70,1.00,1.10,0.90,1.30,1.00,0.80,1.20,0.60,1.00  
    2023-01-01 05:00,0.75,0.95,1.15,0.95,1.35,1.05,0.85,1.25,0.65,1.05  
    2023-01-01 06:00,0.80,0.90,1.20,1.00,1.40,1.10,0.90,1.30,0.70,1.10  
    2023-01-01 07:00,0.85,0.85,1.25,1.05,1.45,1.15,0.95,1.35,0.75,1.15  
    2023-01-01 08:00,0.90,0.80,1.30,1.10,1.50,1.20,1.00,1.40,0.80,1.20  
    2023-01-01 09:00,0.95,0.75,1.35,1.15,1.55,1.25,1.05,1.45,0.85,1.25  
    2023-01-01 10:00,1.00,0.70,1.40,1.20,1.60,1.30,1.10,1.50,0.90,1.30  
    2023-01-01 11:00,1.05,0.65,1.45,1.25,1.65,1.35,1.15,1.55,0.95,1.35  
    2023-01-01 12:00,1.10,0.60,1.50,1.30,1.70,1.40,1.20,1.60,1.00,1.40  
    2023-01-01 13:00,1.15,0.55,1.55,1.35,1.75,1.45,1.25,1.65,1.05,1.45  
    2023-01-01 14:00,1.20,0.50,1.60,1.40,1.80,1.50,1.30,1.70,1.10,1.50  
    2023-01-01 15:00,1.25,0.45,1.65,1.45,1.85,1.55,1.35,1.75,1.15,1.55  
    2023-01-01 16:00,1.30,0.40,1.70,1.50,1.90,1.60,1.40,1.80,1.20,1.60  
    2023-01-01 17:00,1.35,0.35,1.75,1.55,1.95,1.65,1.45,1.85,1.25,1.65  
    2023-01-01 18:00,1.40,0.30,1.80,1.60,2.00,1.70,1.50,1.90,1.30,1.70  
    2023-01-01 19:00,1.45,0.25,1.85,1.65,2.05,1.75,1.55,1.95,1.35,1.75  
    2023-01-01 20:00,1.50,0.20,1.90,1.70,2.10,1.80,1.60,2.00,1.40,1.80  
    2023-01-01 21:00,1.55,0.15,1.95,1.75,2.15,1.85,1.65,2.05,1.45,1.85  
    2023-01-01 22:00,1.60,0.10,2.00,1.80,2.20,1.90,1.70,2.10,1.50,1.90  
    2023-01-01 23:00,1.65,0.05,2.05,1.85,2.25,1.95,1.75,2.15,1.55,1.95  
    2023-01-02 00:00,1.70,0.00,2.10,1.90,2.30,2.00,1.80,2.20,1.60,2.00  
    2023-01-02 01:00,1.75,0.05,2.15,1.95,2.35,2.05,1.85,2.25,1.65,2.05  
    2023-01-02 02:00,1.80,0.10,2.20,2.00,2.40,2.10,1.90,2.30,1.70,2.10  
    2023-01-02 03:00,1.85,0.15,2.25,2.05,2.45,2.15,1.95,2.35,1.75,2.15  
    2023-01-02 04:00,1.90,0.20,2.30,2.10,2.50,2.20,2.00,2.40,1.80,2.20  
    2023-01-02 05:00,1.95,0.25,2.35,2.15,2.55,2.25,2.05,2.45,1.85,2.25  
    2023-01-02 06:00,2.00,0.30,2.40,2.20,2.60,2.30,2.10,2.50,1.90,2.30  
    2023-01-02 07:00,2.05,0.35,2.45,2.25,2.65,2.35,2.15,2.55,1.95,2.35  
    2023-01-02 08:00,2.10,0.40,2.50,2.30,2.70,2.40,2.20,2.60,2.00,2.40  
    2023-01-02 09:00,2.15,0.45,2.55,2.35,2.75,2.45,2.25,2.65,2.05,2.45  
    2023-01-02 10:00,2.20,0.50,2.60,2.40,2.80,2.50,2.30,2.70,2.10,2.50  
    2023-01-02 11:00,2.25,0.55,2.65,2.45,2.85,2.55,2.35,2.75,2.15,2.55  
    2023-01-02 12:00,2.30,0.60,2.70,2.50,2.90,2.60,2.40,2.80,2.20,2.60  
    2023-01-02 13:00,2.35,0.65,2.75,2.55,2.95,2.65,2.45,2.85,2.25,2.65  
    2023-01-02 14:00,2.40,0.70,2.80,2.60,3.00,2.70,2.50,2.90,2.30,2.70  
    2023-01-02 15:00,2.45,0.75,2.85,2.65,3.05,2.75,2.55,2.95,2.35,2.75  
    2023-01-02 16:00,2.50,0.80,2.90,2.70,3.10,2.80,2.60,3.00,2.40,2.80  
    2023-01-02 17:00,2.55,0.85,2.95,2.75,3.15,2.85,2.65,3.05,2.45,2.85  
    2023-01-02 18:00,2.60,0.90,3.00,2.80,3.20,2.90,2.70,3.10,2.50,2.90  
    2023-01-02 19:00,2.65,0.95,3.05,2.85,3.25,2.95,2.75,3.15,2.55,2.95  
    2023-01-02 20:00,2.70,1.00,3.10,2.90,3.30,3.00,2.80,3.20,2.60,3.00  
    2023-01-02 21:00,2.75,1.05,3.15,2.95,3.35,3.05,2.85,3.25,2.65,3.05  
    2023-01-02 22:00,2.80,1.10,3.20,3.00,3.40,3.10,2.90,3.30,2.70,3.10  
    2023-01-02 23:00,2.85,1.15,3.25,3.05,3.45,3.15,2.95,3.35,2.75,3.15  
    2023-01-03 00:00,2.90,1.20,3.30,3.10,3.50,3.20,3.00,3.40,2.80,3.20  
    2023-01-03 01:00,2.95,1.25,3.35,3.15,3.55,3.25,3.05,3.45,2.85,3.25  
    2023-01-03 02:00,3.00,1.30,3.40,3.20,3.60,3.30,3.10,3.50,2.90,3.30  
    2023-01-03 03:00,3.05,1.35,3.45,3.25,3.65,3.35,3.15,3.55,2.95,3.35  
    2023-01-03 04:00,3.10,1.40,3.50,3.30,3.70,3.40,3.20,3.60,3.00,3.40  
    2023-01-03 05:00,3.15,1.45,3.55,3.35,3.75,3.45,3.25,3.65,3.05,3.45  
    2023-01-03 06:00,3.20,1.50,3.60,3.40,3.80,3.50,3.30,3.70,3.10,3.50  
    2023-01-03 07:00,3.25,1.55,3.65,3.45,3.85,3.55,3.35,3.75,3.15,3.55  
    2023-01-03 08:00,3.30,1.60,3.70,3.50,3.90,3.60,3.40,3.80,3.20,3.60  
    2023-01-03 09:00,3.35,1.65,3.75,3.55,3.95,3.65,3.45,3.85,3.25,3.65  
    2023-01-03 10:00,3.40,1.70,3.80,3.60,4.00,3.70,3.50,3.90,3.30,3.70  
    2023-01-03 11:00,3.45,1.75,3.85,3.65,4.05,3.75,3.55,3.95,3.35,3.75  
    2023-01-03 12:00,3.50,1.80,3.90,3.70,4.10,3.80,3.60,4.00,3.40,3.80  
    2023-01-03 13:00,3.55,1.85,3.95,3.75,4.15,3.85,3.65,4.05,3.45,3.85  
    2023-01-03 14:00,3.60,1.90,4.00,3.80,4.20,3.90,3.70,4.10,3.50,3.90  
    2023-01-03 15:00,3.65,1.95,4.05,3.85,4.25,3.95,3.75,4.15,3.55,3.95  
    2023-01-03 16:00,3.70,2.00,4.10,3.90,4.30,4.00,3.80,4.20,3.60,4.00  
    2023-01-03 17:00,3.75,2.05,4.15,3.95,4.35,4.05,3.85,4.25,3.65,4.05  
    2023-01-03 18:00,3.80,2.10,4.20,4.00,4.40,4.10,3.90,4.30,3.70,4.10  
    2023-01-03 19:00,3.85,2.15,4.25,4.05,4.45,4.15,3.95,4.35,3.75,4.15  
    2023-01-03 20:00,3.90,2.20,4.30,4.10,4.50,4.20,4.00,4.40,3.80,4.20  
    2023-01-03 21:00,3.95,2.25,4.35,4.15,4.55,4.25,4.05,4.45,3.85,4.25  
    2023-01-03 22:00,4.00,2.30,4.40,4.20,4.60,4.30,4.10,4.50,3.90,4.30  
    2023-01-03 23:00,4.05,2.35,4.45,4.25,4.65,4.35,4.15,4.55,3.95,4.35  

     if the question is outside the bounds of the RFP, Let the user know answer might be relevant for data provided.
     If not sure, ask the user to provide more information.
     {query}"""},]

    response = client.chat.completions.create(
        model= "o1-preview", #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
    )

    returntxt = response.choices[0].message.content
    return returntxt

if __name__ == "__main__":
    query = "Create a detailed Root cause analysis and find the issue and also see if you get resolution and then create a report to submit?"
    print(processo1text(query))