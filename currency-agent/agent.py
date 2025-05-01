from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService # Or GcsArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.adk.tools import load_artifacts
from google.adk.code_executors import VertexAiCodeExecutor
from google import genai
import google.genai.types as types

# Import the 'datetime' module to work with date and time
import datetime
import os
import requests
import pandas as pd
import subprocess

from opentelemetry.exporter.cloud_logging import CloudLoggingExporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from traceloop.sdk import Traceloop

trace_exporter = CloudTraceSpanExporter()
metrics_exporter = CloudMonitoringMetricsExporter()
logs_exporter = CloudLoggingExporter()

Traceloop.init(
    app_name='ADK Currency Agent',
    exporter=trace_exporter,
    metrics_exporter=metrics_exporter,
    logging_exporter=logs_exporter)

instruction_prompt_ds_v1 = """
  # Guidelines

  **Objective:** Assist the user in achieving their data analysis goals within the context of a Python Colab notebook, **with emphasis on avoiding assumptions and ensuring accuracy.**
  Reaching that goal can involve multiple steps. When you need to generate code, you **don't** need to solve the goal in one go. Only generate the next step at a time.
  Examples of data analysis are - forecasting, prediction, mathematical and statistical calculations.

  **Trustworthiness:** Always include the code in your response. Put it at the end in the section "Code:". This will ensure trust in your output.

  **Code Execution:** All code snippets provided will be executed within the Colab environment.

  **Statefulness:** All code snippets are executed and the variables stays in the environment. You NEVER need to re-initialize variables. You NEVER need to reload files. You NEVER need to re-import libraries.

  **Imported Libraries:** The following libraries are ALREADY imported and should NEVER be imported again:

  ```tool_code
  import io
  import math
  import re
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  import scipy
  ```

  **Output Visibility:** Always print the output of code execution to visualize results, especially for data exploration and analysis. 
  This includes charts and visualizations.
  For example:
    - To look a the shape of a pandas.DataFrame do:
      ```tool_code
      print(df.shape)
      ```
      The output will be presented to you as:
      ```tool_outputs
      (49, 7)

      ```
    - To display the result of a numerical computation:
      ```tool_code
      x = 10 ** 9 - 12 ** 5
      print(f'{{x=}}')
      ```
      The output will be presented to you as:
      ```tool_outputs
      x=999751168

      ```
    - You **never** generate ```tool_outputs yourself.
    - You can then use this output to decide on next steps.
    - Print variables (e.g., `print(f'{{variable=}}')`.
    - Give out the generated code under 'Code:'.

  **No Assumptions:** **Crucially, avoid making assumptions about the nature of the data or column names.** Base findings solely on the data itself. Always use the information obtained from `explore_df` to guide your analysis.

  **Available files:** Only use the files that are available as specified in the list of available files.

  **Data in prompt:** Some queries contain the input data directly in the prompt. You have to parse that data into a pandas DataFrame. ALWAYS parse all the data. NEVER edit the data that are given to you.

  **Answerability:** Some queries may not be answerable with the available data. In those cases, inform the user why you cannot process their query and suggest what type of data would be needed to fulfill their request.

  **WHEN YOU DO PREDICTION / MODEL FITTING, ALWAYS PLOT FITTED LINE AS WELL **


  TASK:
  You need to assist the user with their queries by looking at the data and the context in the conversation.
    You final answer should summarize the code and code execution relavant to the user query.

    You should include all pieces of data to answer the user query, such as the table from code execution results.
    If you cannot answer the question directly, you should follow the guidelines above to generate the next step.
    If the question can be answered directly with writing any code, you should do that.
    If you doesn't have enough data to answer the question, you should ask for clarification from the user.

    You should NEVER install any package on your own like `pip install ...`.
    When plotting trends, you should make sure to sort and order the data by the x-axis.

    NOTE: for pandas pandas.core.series.Series object, you can use .iloc[0] to access the first element rather than assuming it has the integer index 0"
    correct one: predicted_value = prediction.predicted_mean.iloc[0]
    error one: predicted_value = prediction.predicted_mean[0]
    correct one: confidence_interval_lower = confidence_intervals.iloc[0, 0]
    error one: confidence_interval_lower = confidence_intervals[0][0]

"""

client = genai.Client(
    vertexai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

model = os.getenv("GOOGLE_GEMINI_MODEL")

# Get the current date and time
now = datetime.datetime.now()

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Returns:
        dict: A dictionary containing the weather information with a 'status' key ('success' or 'error') and a 'report' key with the weather details if successful, or an 'error_message' if an error occurred.
    """
    if city.lower() == "new york":
        return {"status": "success",
                "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}
    else:
        return {"status": "error",
                "error_message": f"Weather information for '{city}' is not available."}

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Returns:
        dict: A dictionary containing the current time for a specified city information with a 'status' key ('success' or 'error') and a 'report' key with the current time details in a city if successful, or an 'error_message' if an error occurred.
    """
    import datetime
    from zoneinfo import ZoneInfo

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {"status": "error",
                "error_message": f"Sorry, I don't have timezone information for {city}."}

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return {"status": "success",
            "report": f"""The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}"""}

def get_exchange_rate(base_currency: str, target_currency: str) -> float or None:
    """
    Retrieves the current exchange rate between two currencies using the exchangerate-api.com free service.

    This function fetches the exchange rate from the exchangerate-api.com API, which does not require an API key.
    It handles potential API errors and returns None if the exchange rate cannot be retrieved.

    Args:
        base_currency (str): The currency to convert from (e.g., 'USD').
        target_currency (str): The currency to convert to (e.g., 'EUR').

    Returns:
        float or None: The exchange rate as a float if successful, None on failure or if the rate is not found.

    Raises:
        requests.exceptions.RequestException: if there is an issue with the API request.
    """
    base_currency = base_currency.upper()
    target_currency = target_currency.upper()
    api_url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    print(api_url)

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if 'rates' in data and target_currency in data['rates']:
            return float(data['rates'][target_currency])
        else:
            return None  # Indicate that the target currency rate was not found

    except requests.exceptions.RequestException as e:
        print(f"Error fetching exchange rate: {e}") # Log error for debugging
        return None

def get_historical_exchange_rate(base_currency: str, target_currency: str, date_string: str) -> dict or None:
    """
    Retrieves historical exchange rate data for a specific date from the Exchange Rate API.

    Args:
        api_key (str): Your API key for the Exchange Rate API.
        base_currency (str): The three-letter ISO 4217 currency code for the base currency.
        date_string (str): The date for which to retrieve historical data,
                           in "YYYY-MM-DD" format (e.g., "2024-01-20").

    Returns:
        dict: A dictionary containing the historical exchange rate data, or None if the request fails
              or the date string is invalid.
        None:  Error in the API call or invalid date format.

    Raises:
        requests.exceptions.RequestException: If there's an error with the requests library itself.
    """
    api_key = os.getenv("EXCHANGERATE_API_KEY")  
    print("api key:" + api_key)

    # --- Parse the date string ---
    try:
        date_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
    except ValueError:
        print(f"Error: Invalid date format. Please use 'YYYY-MM-DD'. Received: {date_string}")
        return None
    # --- End of date parsing ---

    # Construct the URL using the parsed year, month, and day.
    url = f"https://openexchangerates.org/api/historical/{date_string}.json?app_id={api_key}"
    print(url)

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}. Response text was: {response.text}")
        return None

async def get_exchange_rate_trend(base_currency: str, target_currency: str, start_date: str, end_date: str, tool_context: ToolContext) -> dict:
    """
    Get the trend of exchange rates between two currencies for a given date range. 

    Args:
        base_currency (str): The currency to convert from (e.g., 'USD').
        target_currency (str): The currency to convert to (e.g., 'EUR').
        start_date (str): The start date for the trend in 'YYYY-MM-DD' format.
        end_date (str): The end date for the trend in 'YYYY-MM-DD' format.

    Returns:
        str: A message providing exchange trend or an error message.
    """
    dates = pd.date_range(start=start_date, end=end_date)
    exchange_rates = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        historical_data = get_historical_exchange_rate(base_currency, target_currency, date_str)
        if historical_data and 'rates' in historical_data and target_currency in historical_data['rates']:
            exchange_rates.append(historical_data['rates'][target_currency])
        else:
            exchange_rates.append(None)  # Append None if rate is not available for that date

    df = pd.DataFrame({'Date': dates, 'Exchange Rate': exchange_rates})
    df.dropna(subset=['Exchange Rate'], inplace=True) # Remove rows with None exchange rates

    if df.empty:
        return "Could not retrieve exchange rate data for the specified date range."
    else:
        return f"Exchange rate trend data is \n {df}"

code_agent = Agent(
    model=os.getenv("GOOGLE_GEMINI_MODEL"),
    name="data_science_agent",
    instruction=instruction_prompt_ds_v1,
    code_executor=VertexAiCodeExecutor(
        optimize_data_file=True,
        stateful=True,
    ),
)

async def call_code_agent(
    question: str,
    tool_context: ToolContext,
):
    """
    Perform data analysis and calculations for the user.
    Examples are - ML algorithms like forecasting, prediction, & mathematical and statistical calculations.

    Args:
        question (str): the input from user and calls nl2py agent

    Returns:
        str: A message providing output of the nl2py agent or an error message.
    """

    question_with_data = f"""
        Question to answer: {question}

        Actual data to analyze previous question is already in the following:
  
     """

    agent_tool = AgentTool(agent=code_agent)

    code_agent_output = await agent_tool.run_async(
        args={"request": question_with_data}, tool_context=tool_context
    )
    tool_context.state["code_agent_output"] = code_agent_output
    return code_agent_output

def call_local_tool(
    question: str,
    tool_context: ToolContext,
):

    """"            
    Local tool can be used to run operating system commands to find cpu, memory, disk, network and process details

    Args:
        question (str): the input from user that is passed to local command tool. 
    Returns:
        str: A message providing output of the call local tool or an error message.

    """

    question_with_prompt = f"""Provide only the command for this {question} so that it can be executed in a shell. Do not use sudo or prefix with bash. """

    contents = [
        types.Content(
            role="user",
            parts=[
            types.Part.from_text(text=question_with_prompt)
            ]
        ),
    ]
    tools = []
    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        max_output_tokens = 8192,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        tools = tools,
    )

    call_local_llm_output = ""
    for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        print(chunk.text, end="")
        call_local_llm_output = call_local_llm_output + chunk.text

    call_local_llm_output = call_local_llm_output.replace("`","")
    print("call_local_llm_output:", call_local_llm_output)
    process = subprocess.Popen(call_local_llm_output, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    tool_context.state["call_local_llm_output"] = call_local_llm_output

    call_local_tool_output = ""
    if stderr:
        print(f"Error: {stderr.decode()}")
    else:
        call_local_tool_output = stdout.decode()

    tool_context.state["call_local_tool_output"] = call_local_tool_output
    print("call_local_tool_output:", call_local_tool_output)
    return call_local_tool_output

def call_web_search(
    question: str,
    tool_context: ToolContext,
):

    """
    Args:
        question (str): the input from user that calls web search tool. 
        Web search tool can be used to find news, current events any other information that is not provided by other tools.

    Returns:
        str: A message providing output of the web search tool or an error message.
    """

    question_with_data = f"""
        Question to answer: {question}

    """
    print("question_with_data:", question)

    contents = [
        types.Content(
            role="user",
            parts=[
            types.Part.from_text(text=question_with_data)
            ]
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 0.95,
        max_output_tokens = 8192,
        response_modalities = ["TEXT"],
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        tools = tools,
    )

    web_search_tool_output = ""
    for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        print(chunk.text, end="")
        web_search_tool_output = web_search_tool_output + chunk.text
        
    tool_context.state["web_search_tool_output"] = web_search_tool_output
    print("web_search_tool_output:", web_search_tool_output)
    return web_search_tool_output

root_agent = Agent(
    name="exchangerate_weather_time_agent",
    model=os.getenv("GOOGLE_GEMINI_MODEL"),
    description="Agent to answer questions about the historical and current exchange rates, create and run python code, search the web to get information and time and weather in a city. Today is " + now.strftime("%Y-%m-%d %H:%M:%S"),
    instruction="I can answer your questions about the historical and current currency exchange rates, create and run python code, search the web to get information and time and weather in a city and Today is " + now.strftime("%Y-%m-%d %H:%M:%S") + ". Always format the user response in markdown format.",
    tools=[get_weather, get_current_time, get_exchange_rate, get_historical_exchange_rate, get_exchange_rate_trend, load_artifacts, call_code_agent, call_web_search, call_local_tool],
)

# Instantiate the desired artifact service and session service
artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name="Currency Agent",
    session_service=session_service,
    artifact_service=artifact_service,
)

session = session_service.create_session(app_name="Currency Agent", user_id='rajan',)
