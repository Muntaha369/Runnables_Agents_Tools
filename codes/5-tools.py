from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage
from rich import print
from langchain_openrouter import ChatOpenRouter

load_dotenv()

llm = ChatOpenRouter( model = "gpt-4o-mini" )

#1.Creating a tool

@tool
def findlen(text:str) -> int:
    """Give the length of the text"""
    return len(text)

#2.Tool binding

llm_tool = llm.bind_tools([findlen])

#Here beleow tool is not invoke and tools_calls array is empty so res2 will respond like normal chatbot

# res1 = llm.invoke("hello there")
# res2 = llm_tool.invoke("hello there")

# print(res1)
# print(res2)

#Here below tool is invoke and tools_calls array is also not empty so response2 will respond using the tool it has but you need to invoke it

res1 = llm.invoke("How many characters are in the word elephant?")

res2 = llm_tool.invoke(
    "How many characters are in the word elephant?"
)

print(res1)
print("===Diff===")

#3. Tool calling
print(res2.tool_calls[0]) 


#Go to 6th file for tool execution