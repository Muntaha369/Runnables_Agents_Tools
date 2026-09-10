#Runnable passthrough is used to get the output in middle of a pipeline

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenRouter(
    model  = "gpt-4o-mini"
)

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system","You are a codegenerator you only generete code and thats it no explanation whatsoever"),
    ("human", "{topic}")
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system","you summarize the code "),
    ("human","{code}")
])

pass1 = code_prompt | model | parser

# RunnableParallel : Lets you run stuff parallely and also lets you store output in middle of pipeline thorugh RunnablePassthrough
pass2 = RunnableParallel({
    "code":RunnablePassthrough(),
    "explanation": summarize_prompt | model | parser
})

chain = pass1 | pass2 # type: ignore

result = chain.invoke({"topic":"Write a code for 2sum"})

print("===CODE===")
print(result['code'])

print("===Summary===")
print(result['explanation'])