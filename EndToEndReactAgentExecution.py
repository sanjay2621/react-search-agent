"""
Detailed step‑by‑step explanation of EndToEndReactAgentExecution.py

Imports
typing.List, Union: Type hints for lists and a union of possible agent outputs.
dotenv.load_dotenv: Loads environment variables (e.g., OPENAI_API_KEY) from a .env file into the process environment.
ReActSingleInputOutputParser: Parses a single ReAct‑style LLM output into either an AgentAction or AgentFinish.
AgentAction, AgentFinish: LangChain core data classes representing an intermediate tool call or a final answer.
PromptTemplate: For templating the prompt with dynamic values.
Tool, render_text_description, tool: Constructs and registers tools for the agent; @tool wraps a function into a LangChain Tool.
ChatOpenAI: LangChain wrapper for OpenAI chat models.

Environment loading
load_dotenv() searches for a .env file and loads key/value pairs into os.environ. Required so ChatOpenAI can authenticate.

Tool definition
@tool decorator converts get_text_length into a LangChain Tool with:
name: auto‑derived (get_text_length).
description: docstring "Returns the length of a text by characters".

Function logic:
Debug print of the raw input.
Strips surrounding quotes and newlines (defensive cleanup).
Returns len(text).

Helper find_tool_by_name
Linear search over a list of Tool objects to match by tool.name.
Raises ValueError if not found. Used to map the model’s chosen tool name to the actual callable.

__main__ guard
Ensures the contained logic only runs when the script is executed directly, not when imported.

Startup log
print("Hello ReAct LangChain!") for simple runtime confirmation.

Tool collection
tools = [get_text_length] creates the list of available tools for the agent.

Prompt template string
Defines a ReAct instruction pattern:
Lists available tools.
Specifies the required formatting: alternating Thought, Action, Action Input, Observation.

Terminates with Final Answer.
Ends with an initial Question: {input}\nThought: so the model must produce the first Thought:.

PromptTemplate construction
PromptTemplate.from_template(template=template) parses the template.
.partial(...) pre‑fills:
tools: Rendered tool descriptions via render_text_description(tools) (includes name + description).
tool_names: Comma‑joined list of tool names for the Action constraint.
Result: At runtime only {input} must be supplied.

LLM initialization
ChatOpenAI(model="gpt-4", temperature=0, stop=["\nObservation", "Observation"])
temperature=0 for deterministic reasoning.
stop ensures the model output stops before (or at) an unwanted trailing Observation token, allowing parser control.
Model expected to produce at least Thought: and optionally an Action.

intermediate_steps placeholder

Created but unused. In a full agent loop this would accumulate (AgentAction, observation) pairs for iterative reasoning.

Chain (agent) assembly
Dictionary {"input": lambda x: x["input"]}: Simple input mapping for the pipeline.
Piped (|) into:
prompt: Fills prompt with the provided input.
llm: Generates a ReAct‑formatted response.
ReActSingleInputOutputParser(): Parses the model text into:
AgentAction(tool=..., tool_input=..., log=...) if an Action is emitted.
AgentFinish(return_values={"output": ...}, log=...) if a Final Answer is produced directly.

Invocation
agent.invoke({"input": "What is the length of 'DOG' in characters?"})

Model likely responds:
Thought: Need to use length tool
Action: get_text_length
Action Input: 'DOG'
Parser transforms that into an AgentAction.
Output logging
print(agent_step) shows the structured AgentAction or AgentFinish.

Tool execution branch
if isinstance(agent_step, AgentAction):
Extracts tool_name and finds the tool via find_tool_by_name.
Gets tool_input.

Calls tool_to_use.func(str(tool_input)):
Note: tool_input may already be a string; wrapping in str() is defensive.
Prints observation=... (e.g., 3).

End of script
No loop to feed the observation back. This is a single‑step manual execution. A full ReAct agent would:
Append (agent_step, observation) to intermediate_steps.
Re‑prompt the model including the prior steps until a Final Answer is generated.
Key behaviors and considerations
The script demonstrates only the first reasoning/action step; it does not complete the ReAct loop.
stop tokens may truncate outputs if the model includes Observation prematurely; acceptable here because manual tool invocation follows.
Stripping quotes in get_text_length can alter intentional surrounding quotes; acceptable for this controlled example.
Absence of error handling for malformed model output (the parser would raise).
Possible improvements (not executed)
Implement iterative loop to handle multiple tool calls.
Pass intermediate_steps back into a prompt template.
Add logging or structured tracing instead of print.
Validate tool input before execution.
Let me know if you need a loop version or enhancements.
"""

from typing import List, Union

from dotenv import load_dotenv
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI

import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non-alphabetic characters just in case

    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    You have access to the following tools:
    {tools}
    
    Use the following format EXACTLY:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT: 
    - For "Action:" line, output ONLY the exact tool name from this list: [{tool_names}]
    - Do NOT write explanatory text on the Action line
    - The Action must be exactly one of: {tool_names}

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler.AgentCallbackHandler()],
    )
    intermediate_steps = []
    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
            }
            | prompt
            | llm
            | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(agent_step)
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)

## CALL BACK HANDLER RESPONSE - AGENTCALLBACKHANDLER.PY
"""
Hello ReAct LangChain!
***Prompt to LLM was:***
Human: 
    You have access to the following tools:
    get_text_length(text: str) -> int - Returns the length of a text by characters
    
    Use the following format EXACTLY:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [get_text_length]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT: 
    - For "Action:" line, output ONLY the exact tool name from this list: [get_text_length]
    - Do NOT write explanatory text on the Action line
    - The Action must be exactly one of: get_text_length

    Begin!

    Question: What is the length of the word: DOG
    Thought: 
    
*********
***LLM Response:***
I need to find the length of the word "DOG". I can use the get_text_length tool to find this.
    Action: get_text_length
    Action Input: DOG
    
*********
tool='get_text_length' tool_input='DOG\n' log='I need to find the length of the word "DOG". I can use the get_text_length tool to find this.\n    Action: get_text_length\n    Action Input: DOG\n    '
get_text_length enter with text='DOG\n'
observation=3
***Prompt to LLM was:***
Human: 
    You have access to the following tools:
    get_text_length(text: str) -> int - Returns the length of a text by characters
    
    Use the following format EXACTLY:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [get_text_length]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT: 
    - For "Action:" line, output ONLY the exact tool name from this list: [get_text_length]
    - Do NOT write explanatory text on the Action line
    - The Action must be exactly one of: get_text_length

    Begin!

    Question: What is the length of the word: DOG
    Thought: I need to find the length of the word "DOG". I can use the get_text_length tool to find this.
    Action: get_text_length
    Action Input: DOG
    
Observation: 3
Thought: 
    
*********
***LLM Response:***
I now know the final answer
Final Answer: 3
*********
return_values={'output': '3'} log='I now know the final answer\nFinal Answer: 3'
### AgentFinish ###
{'output': '3'}

Process finished with exit code 0
"""