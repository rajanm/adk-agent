# ADK Agent Examples

## Currency Agent

## Description

The ADK Currency Agent is a sample agent built using the Google Agent Development Kit (ADK). It allows users to perform currency conversions, get the current time, get weather information, and leverage data science capabilities through a data science agent and web search functionality. This agent is designed to be deployed on Google Cloud Run.

## Key Files

*   `main.py`: This is the main application file that defines the FastAPI app and integrates the agent logic using the `google.adk.cli.fast_api` module.
*   `currency-agent/`: This directory contains the core logic of the currency agent, including the agent definition, actions, and models.
*   `deploy.sh`: This script is used to deploy the agent to Google Cloud Run. It sets the necessary environment variables and deploys the application from the current directory.
*   `Dockerfile`: This file defines the Docker image for the agent, specifying the base image, dependencies, and startup command.
*   `requirements.txt`: This file lists the Python dependencies required to run the agent.

## Setup

1.  **Developer Setup:**
    In case you don't have it yet, create an account in Google Cloud. Activate the Cloud Shell and Cloud Editor.

    a. Once the shell is activated, install uv using "pip install uv". 
    b. Create a folder called adk-agent. 
    c. And, inside this folder, create a folder called "currency-agent". Within this folder, run "uv init".
    d. uv will automatically create a virtual environment called ".venv".
    e. The virtual environment can be activated using "source .venv/bin/activate".

2.  **Set up environment variables:**

    Create a `setenv.sh` file in the root directory of the project and set the following environment variables:

    ```
    # Use GOOGLE_API_KEY Key if GOOGLE_GENAI_USE_VERTEXAI=False
    # Otherwise, provide GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION details
    # export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    
    export GOOGLE_CLOUD_PROJECT="YOUR_GOOGLE_CLOUD_PROJECT"
    export GOOGLE_CLOUD_LOCATION="us-central1" # Or your preferred location
    export GOOGLE_GENAI_USE_VERTEXAI=True
    export GOOGLE_GEMINI_MODEL="gemini-2.0-flash-001"

    # Create a API key at https://www.exchangerate-api.com/
    export EXCHANGERATE_API_KEY="YOUR_EXCHANGERATE_API_KEY"

    export SERVICE_NAME="adk-currency-agent-service"
    export APP_NAME="adk-currency-agent-app"
    export AGENT_PATH="./currency-agent"

    # This can be empty. Once the extension is created by code, you can add it here
    export CODE_INTERPRETER_EXTENSION_NAME="projects/YOUR_PROJECT_ID/locations/YOUR_LOCATION/extensions/YOUR_EXTENSION_ID"
    ```

    Replace the placeholder values with your actual API keys and project details. Give execute permissions to the setenv.sh file and execute "source setenv.sh" to set the environment variables.

2.  **Install dependencies:**

    Run the following command to install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running

To run the agent locally, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

Alternatively, you can use the `run.sh` script. Ensure that this script also has execute permissions.

```bash
./run.sh
```

In the Google Cloud console, select the "Preview" option. This will open the ADK Web Developer UI for testing.

## Deployment

To deploy the agent to Google Cloud Run, follow these steps:

1.  **Set the environment variables:**

    Make sure you have set the environment variables in your shell or using the `setenv.sh` script:

    ```bash
    source setenv.sh
    ```

2.  **Deploy to Cloud Run:**

    Run the `deploy.sh` script:

    ```bash
    ./deploy.sh
    ```

    This script will deploy the agent to Google Cloud Run, making it accessible over the internet.
    Once you have completed testing, you may delete the Cloud Run service.