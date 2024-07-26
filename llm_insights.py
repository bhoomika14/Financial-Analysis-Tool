from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import os
import json
import streamlit as st

with open('config.json') as f:
    config = json.load(f)
os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0)

@st.cache_resource
def llm_data(data):
    insights = llm.invoke(f"""{data} is a financial data. The data represents the monthly sales of medicines for the past several years. 
                          The data includes DATE column and AMOUNT column. Analyse the data and provide your insights into the following aspects of this data:<br>
                          1. Seasonal Trends: What seasonal patterns or trends are visible in the sales data?

                          2. Trend Analysis: What are the long-term trends in the sales data? Are sales generally increasing, decreasing, or remaining stable over time?

                          3. Anomalies and Outliers: Are there any significant anomalies or outliers in the sales data?

                          4. Forecasting: Based on historical data, what are the projected sales figures for the upcoming months or years? Are there any forecasts or predictions that can be made?</br>
                          """)

    return insights.content