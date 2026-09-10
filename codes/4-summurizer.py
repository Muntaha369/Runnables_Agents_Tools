from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
load_dotenv()

model = ChatOpenRouter( model="gpt-4o-mini" )

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Summarize the news in distinct points.

    {news}
""")

tool = TavilySearch(
    max_results=5,
)

chain = prompt | model | parser

news_result = tool.invoke("Today's latest news")

result = chain.invoke({"news": news_result}) # type: ignore

print(result)