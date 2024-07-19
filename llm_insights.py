from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json

with open('config.json') as f:
    config = json.load(f)
os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def llm_data(data):
    insights = llm.invoke(f"{data} is a time series data. Provide your insights by analyzing the data.")

    return insights.content