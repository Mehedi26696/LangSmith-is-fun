
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    api_key=api_key,
)  
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of France?"})
print(result)