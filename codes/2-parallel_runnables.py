#Parallel runnables is use to execute a pipeline/chian simuntaneously

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

load_dotenv()

model = ChatOpenRouter(model="gpt-4o-mini")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# RunnableParallel : Lets you run stuff parallely and also lets you store output in middle of pipeline thorugh RunnablePassthrough
chain = RunnableParallel({
    #The lambda insde is used to get the value of short or long and x represent a variable which is actually storing them
    "short" :RunnableLambda(lambda x :x['short']) |short_prompt | model | parser , # type: ignore
    "long":RunnableLambda(lambda x : x['long']) | long_prompt | model | parser # type: ignore
})

result = chain.invoke({
    "short":{"topic":"ML"},
    "long":{"topic":"DL"}
})

print(result['short'])
print('\n===LongNow===')
print(result['long'])