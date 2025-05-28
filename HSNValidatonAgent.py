import pandas as pd
import google.generativeai as genai
import google.generativeai.types as types
import asyncio
import inspect
import json
import os
from fastapi import FastAPI, Request
from google.adk.agent import Agent, AgentService, AgentCallbacks, SimpleAgentCallbacks
from google.adk.agents import GeminiAgent
from google.adk.integrations.fastapi import AgentRouter
from google.adk.io import EventHandler
from google.adk.io.v1 import AgentOutput, Event
from google.adk.runners import AgentRunner, AppRunner
from google.adk.sessions import InMemorySessionService

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. Please set it before running the agent."
    )
genai.configure(api_key=GEMINI_API_KEY)
print("DEBUG: Google Generative AI client configured from environment variable.")

# --- 1. Load Master HSN Data ---
def load_hsn_data(file_path: str = "HSN_Master_Data.csv") -> pd.DataFrame:
    """Loads HSN master data from a CSV file with ISO-8859-1 encoding."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip()
        df["HSNCode"] = df["HSNCode"].astype(str).str.strip()
        df.set_index("HSNCode", inplace=True)
        print(f"Successfully loaded {len(df)} HSN codes from {file_path}")
        return df
    except FileNotFoundError:
        print(f"ERROR: HSN_Master_Data.csv not found at {file_path}. Please ensure it's in the same directory.")
        return pd.DataFrame()

# Load data globally once
HSN_MASTER_DATA = load_hsn_data()

# --- 2. Define the HSN Validation Tools (Python functions) ---
# Validates a single HSN code against the master dataset and returns its description if valid.

def validate_hsn_code(hsn_code: str) -> dict:

    if HSN_MASTER_DATA.empty:
        return {
            "status": "error",
            "message": "HSN master data not loaded. Please ensure 'HSN_Master_Data.csv' is available.",
        }

    hsn_code = str(hsn_code).strip()

    if not 2 <= len(hsn_code) <= 8:
        return {
            "status": "invalid",
            "hsn_code": hsn_code,
            "message": "HSN code must be between 2 and 8 digits.",
        }

    if hsn_code in HSN_MASTER_DATA.index:
        description = HSN_MASTER_DATA.loc[hsn_code, "Description"]
        return {"status": "valid", "hsn_code": hsn_code, "description": description}
    else:
        return {
            "status": "invalid",
            "hsn_code": hsn_code,
            "message": "HSN code not found in master data.",
        }

# Validates multiple HSN codes against the master dataset and returns its description if valid.
def validate_multiple_hsn_codes(hsn_codes: list[str]) -> list[dict]:
    results = []
    for code in hsn_codes:
        results.append(validate_hsn_code(code))
    return results

# Validates the given text input against the master dataset (description column) and returns its code/description if valid.
def lookup_by_description(search_text: str) -> list[str]:
    if HSN_MASTER_DATA.empty:
        return ["Error: HSN master data not loaded. Please ensure 'HSN_Master_Data.csv' is available."]

    search_text = search_text.strip().lower()

    temp_df = HSN_MASTER_DATA.reset_index()

    matches = temp_df[temp_df['Description'].str.lower().str.contains(search_text, na=False)]

    if matches.empty:
        return [f"No HSN codes found for descriptions matching: '{search_text}'"]

    results = []
    for _, row in matches.iterrows():
        results.append(f"{row['HSNCode']}: {row['Description']}")

    return results

# --- 3. Define the Agent ---
class HSNValidationAgent(Agent):
    def __init__(self):
        super().__init__()
        self._gemini_agent = GeminiAgent(
            model_name="gemini-1.5-flash",
            tools=[
                validate_hsn_code,
                validate_multiple_hsn_codes,
                lookup_by_description,
            ],
            system_instruction="""
            You are an HSN Code validation and lookup agent. Your primary role is to:
            1.  Validate HSN codes provided by the user against a master dataset and provide their descriptions.
            2.  Search for HSN codes based on descriptive keywords.

            Here's how you should operate:
            - If the user provides an HSN code (or a list of HSN codes) for validation:
                - Use `validate_hsn_code` for a single HSN code.
                - Use `validate_multiple_hsn_codes` for a list of HSN codes.
                - Before validating against the master data, ensure the HSN code is between 2 and 8 digits long. If not, inform the user it's an invalid HSN code.
                - If the HSN code is valid (length check passed and found in master data), return the HSN code and its description.
                - If the HSN code is invalid (either length check fails or not found in master data), clearly state the reason.
            - If the user asks to find HSN codes by a descriptive term (e.g., "find HSN for live animals", "what HSN is for vegetable oil"):
                - Use the `lookup_by_description` tool.
                - Present the results clearly, listing each matching HSN code and its description.
            - Be polite and helpful.
            - If the user asks for something that is clearly not an HSN code related query (e.g., "What is the weather?"), explain that you are an HSN validation and lookup agent and can only help with HSN codes and descriptions.
            """
        )

    async def _handle_message(self, event: Event, event_handler: EventHandler):
        async for agent_output in self._gemini_agent.handle_event(event):
            yield agent_output

# --- 4. Run the Agent as a FastAPI Service ---
app = FastAPI()
agent_router = AgentRouter(
    agent=HSNValidationAgent(),
    session_service=InMemorySessionService(), # This is the service causing the 'Session not found' bug in ADK 1.0.0
    agent_id="hsn_validation_agent",
    project_id="default",
    location_id="default",
    app=app,
)

if __name__ == "__main__":
    # If running with `python HSNValidatonAgent.py`, Uvicorn will be started.
    # The 'Session not found' error persists due to an ADK bug with InMemorySessionService.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
