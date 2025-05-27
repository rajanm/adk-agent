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
import datetime # Still needed for `now`
import os # Still needed for getenv
import subprocess # Still needed for call_local_tool
# requests and pandas are no longer directly used in this file, moved to tools.py

from .tools import (
    get_weather,
    get_current_time,
    get_exchange_rate,
    get_historical_exchange_rate,
    get_exchange_rate_trend
)

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

# Functions get_weather, get_current_time, get_exchange_rate, 
# get_historical_exchange_rate, get_exchange_rate_trend are now imported from .tools

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
    """
    Local tool to run predefined, safe operating system commands to get system information.

    **Security Warning:** This tool is designed to execute shell commands based on LLM interpretation
    of user input. It is restricted to a predefined list of safe, informational commands
    to minimize security risks. Arbitrary command execution is not permitted.

    Args:
        question (str): The user's question, which will be mapped to an allowed command.
    Returns:
        str: The output of the executed command if successful and allowed, or an error message.
    """

    ALLOWED_COMMANDS = {
        "df -h": "Show disk space usage.",
        "free -m": "Show memory usage in megabytes.",
        "uptime": "Show system uptime and load.",
        "vmstat": "Show virtual memory statistics.",
        "iostat": "Show CPU and I/O statistics.",
        "netstat -tulnp": "Show active network connections (TCP/UDP, listening, numeric ports, process IDs). Note: Requires 'net-tools' package.",
        # "top -bn1": "Show current running processes. Note: Can be resource-intensive." # Example of a command that might be too verbose or resource-intensive
    }

    allowed_commands_formatted = "\n".join([f"- '{cmd}' (for: {desc})" for cmd, desc in ALLOWED_COMMANDS.items()])
    question_with_prompt = f"""
    Given the user's question: '{question}'
    Select the most appropriate command from the following list to answer the question.
    Output ONLY the command string itself (e.g., 'df -h').
    If none of the commands are suitable for answering the question, output 'UNKNOWN'.

    Available commands:
    {allowed_commands_formatted}
    """

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

    call_local_llm_output = call_local_llm_output.replace("`","").strip()
    print("call_local_llm_output (raw):", call_local_llm_output)

    tool_context.state["call_local_llm_output_raw"] = call_local_llm_output
    call_local_tool_output = ""

    if call_local_llm_output in ALLOWED_COMMANDS:
        command_to_execute = call_local_llm_output
        command_parts = command_to_execute.split() # Split command for shell=False
        print(f"Executing allowed command: {command_to_execute}")
        try:
            # Using shell=False is safer as it avoids shell injection if command_parts are properly sanitized (which they are, by coming from a fixed list)
            process = subprocess.Popen(command_parts, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if stderr:
                print(f"Error executing command '{command_to_execute}': {stderr}")
                call_local_tool_output = f"Error executing command: {stderr}"
            else:
                call_local_tool_output = stdout
                print(f"Command '{command_to_execute}' output: {stdout}")

        except FileNotFoundError:
            print(f"Error: Command '{command_parts[0]}' not found. Ensure it is installed and in PATH.")
            call_local_tool_output = f"Error: Command '{command_parts[0]}' not found. It might need to be installed."
        except Exception as e:
            print(f"An unexpected error occurred while executing '{command_to_execute}': {e}")
            call_local_tool_output = f"An unexpected error occurred: {e}"
    elif call_local_llm_output == "UNKNOWN":
        print("LLM determined no suitable command.")
        call_local_tool_output = "Sorry, I could not find a suitable command to answer your question from the allowed list."
    else:
        print(f"Command '{call_local_llm_output}' is not in ALLOWED_COMMANDS or is 'UNKNOWN'. Not executing.")
        call_local_tool_output = "Sorry, I can only execute a predefined set of informational commands, and your request did not map to one of them or the command was not recognized."

    tool_context.state["call_local_tool_output"] = call_local_tool_output
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
    tools=[
        get_weather, 
        get_current_time, 
        get_exchange_rate, 
        get_historical_exchange_rate, 
        get_exchange_rate_trend,
        load_artifacts, 
        call_code_agent, 
        call_web_search, 
        call_local_tool
    ],
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
