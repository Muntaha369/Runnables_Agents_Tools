from dotenv import load_dotenv
import os
from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent 
from langchain.agents.middleware import wrap_tool_call

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Creating a city news tool 
@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    #Get result and stores in response
    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    #gets result
    results = response.get("results", [])

    #handaling case if there is no news regaurding that city
    if not results:
        return f"No news found for {city}"
    
    news_list = []

    #storing all the results in array including title url and content
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        
        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    #returning that to agent
    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)

#llm Setup


llm = ChatOpenRouter(model="gpt-4o-mini") #type:ignore

#creating a middleware beteen agent and the tool
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)  

agent = create_agent(
    llm,
    tools = [get_news],
    system_prompt= "you are a helpful city assistant.",
    middleware= [human_approval]
)

print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })

    print("bot : ", result['messages'][-1].content )