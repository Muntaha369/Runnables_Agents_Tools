#Running Runnables sequentially shorthens the length of code

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenRouter(model="gpt-4o-mini")

parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

#This is what considered as a simple runnable
pipeline = short_prompt | model | parser 

result = pipeline.invoke({"topic":"ML"})

print(result)