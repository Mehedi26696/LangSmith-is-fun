from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Set Langsmith project for tracing
os.environ['LANGSMITH_PROJECT'] = 'Sequential Chain Demo'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

api_key = os.getenv("GEMINI_API_KEY")
model1 = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    temperature=0.7,
    api_key=api_key,
)

model2 = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    temperature=0.5,
    api_key=api_key,
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# Invoke with tracing config
config = {
    'run_name': 'Sequential Chain Run',
    'tags': ['sequential', 'chain', 'demo', 'langsmith', "report", "summary"],
    'metadata': {'project': 'Sequential Chain Demo', 'model1': 'gemini-2.5-flash', 'model2': 'gemini-2.5-flash', 'model1_temperature': 0.7, 'model2_temperature': 0.5}
}

result = chain.invoke({'topic': 'Unemployment in Bangladesh'}, config=config)

print(result)