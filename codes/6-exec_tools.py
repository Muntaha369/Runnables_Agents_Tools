# explanation https://chatgpt.com/s/t_6aa2f6ed8cf48191bde33f5a0e3dc592

from dotenv import load_dotenv
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage
from rich import print
from langchain_openrouter import ChatOpenRouter

load_dotenv()


# Create the LLM
llm = ChatOpenRouter(
    model="gpt-4o-mini"
)


# ============================================================
# 1. CREATE A TOOL
# ============================================================

# @tool converts this normal Python function into a
# LangChain tool that the LLM can know about and request.
#
# IMPORTANT:
# The LLM does NOT execute this function itself.
# Our Python program will execute it later.

@tool
def findlen(text: str) -> int:
    """Give the length of the text"""
    return len(text)


# ============================================================
# 2. BIND THE TOOL TO THE LLM
# ============================================================

# Tell the LLM:
# "You have access to this tool. You can request it when needed."
#
# IMPORTANT:
# bind_tools() does NOT execute the tool.
# It only makes the tool available to the LLM.

llm_tool = llm.bind_tools([findlen])


# ============================================================
# 3. CREATE CONVERSATION HISTORY
# ============================================================

# We store all messages in this list.
#
# The conversation will eventually look like:
#
# HumanMessage → User's question
# AIMessage    → LLM requests a tool
# ToolMessage  → Tool returns its result
#
# Keeping these messages is important because we send the
# conversation back to the LLM after executing the tool.

messages = [
    HumanMessage(
        content="How many characters are in the word elephant?"
    )
]


# ============================================================
# 4. FIRST LLM CALL — TOOL CALLING
# ============================================================

# The LLM receives:
#
#   User question
#       +
#   Available tool: findlen
#
# The LLM decides whether it needs to use the tool.
#
# IMPORTANT:
# At this point the tool has NOT been executed yet.
#
# The LLM only returns a request such as:
#
#   "Call findlen with text='elephant'"

res = llm_tool.invoke(messages)

print("=== LLM TOOL CALL ===")
print(res.tool_calls)


# ============================================================
# 5. ADD THE LLM RESPONSE TO THE CONVERSATION
# ============================================================

# The LLM's response contains its tool-call request.
#
# We add it to the conversation so that the LLM will remember
# what it requested when we call it again later.

messages.append(res) # type:ignore


# ============================================================
# 6. EXECUTE THE TOOL
# ============================================================

# Get the first tool call requested by the LLM.
#
# Example:
#
# {
#     "name": "findlen",
#     "args": {
#         "text": "elephant"
#     }
# }

tool_call = res.tool_calls[0]


# NOW the Python program actually executes the tool.
#
# This is different from bind_tools().
#
# bind_tools() → tells the LLM about the tool
# invoke()     → actually runs the Python tool
#
# Internally this is basically:
#
# findlen("elephant")
#
# which returns:
#
# 8

tool_result = findlen.invoke(tool_call)

print("=== TOOL RESULT ===")
print(tool_result)


# ============================================================
# 7. ADD TOOL RESULT TO THE CONVERSATION
# ============================================================

# The LLM needs to know what the tool returned.
#
# So we add the ToolMessage to our conversation history.
#
# Our messages now look like:
#
# HumanMessage → "How many characters..."
# AIMessage    → "Call findlen('elephant')"
# ToolMessage  → "8"

messages.append(tool_result)


# ============================================================
# 8. SECOND LLM CALL — FINAL ANSWER
# ============================================================

# Send the complete conversation back to the LLM.
#
# The LLM can now see:
#
#   User: How many characters are in elephant?
#   AI:   I want to call findlen("elephant")
#   Tool: 8
#
# Now the LLM has the tool result and can generate
# a normal final response.

final_result = llm_tool.invoke(messages)


# ============================================================
# 9. PRINT THE FINAL ANSWER
# ============================================================

print("=== FINAL ANSWER ===")
print(final_result.content)

print("=== Messages ===")
print(messages)