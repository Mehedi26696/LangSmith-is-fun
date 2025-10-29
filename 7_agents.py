from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
import os
from langchain.agents import create_agent  # note: AgentState is *not* used like below
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "Agents"

gemini_api_key = os.getenv("GEMINI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

search_tool = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city."""
    url = f'https://api.weatherstack.com/current?access_key={weather_api_key}&query={city}'
    response = requests.get(url)
    return response.json()

system_prompt = """
You are a reasoning agent that can use tools.
Whenever you need information, think, then use a tool, then observe the output.
Always respond with reasoning steps clearly.
"""

agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=system_prompt
)
 
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Find the birth city of the current captain of the Liverpool football club and get the current weather there."}]}
)

print("Final Output:", response)
