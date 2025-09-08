"""
Here’s a simple explanation for each line in your code, with examples:

1. `from dotenv import load_dotenv`
   Imports a function to load environment variables from a `.env` file.
   *Example:* You store your API keys in `.env` and load them securely.

2. `from langchain import hub`
   Imports the `hub` module from LangChain, used to get pre-made prompts.
   *Example:* You fetch a prompt template for your agent.

3. `from langchain.agents import AgentExecutor`
   Imports a class to run agents with tools.
   *Example:* You use this to execute your search agent.

4. `from langchain.agents.react.agent import create_react_agent`
   Imports a function to create a "ReAct" agent (an agent that reasons and acts).
   *Example:* You build an agent that can search and answer questions.

5. `from langchain_openai import ChatOpenAI`
   Imports a class to use OpenAI’s chat models.
   *Example:* You use GPT-4 for your agent’s responses.

6. `from langchain_tavily import TavilySearch`
   Imports a tool for web search using Tavily.
   *Example:* Your agent can search the web for answers.

7. `load_dotenv()`
   Loads environment variables from `.env` into your program.
   *Example:* Your OpenAI API key is now available.

8. `tool = [TavilySearch()]`
   Creates a list with the Tavily search tool.
   *Example:* The agent can use this tool to search online.

9. `llm = ChatOpenAI(model="gpt-4", temperature=0)`
   Sets up the GPT-4 model with no randomness.
   *Example:* The agent gives consistent answers.

10. `react_prompt = hub.pull("hwchase17/react")`
    Gets a pre-made prompt for the ReAct agent from the hub.
    *Example:* The agent knows how to reason and act.

11. `agent = create_react_agent(llm=llm, tools=tool, prompt=react_prompt)`
    Creates the ReAct agent with the model, tools, and prompt.
    *Example:* The agent is ready to answer questions using search.

12. `agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)`
    Sets up the agent executor to run the agent and show details.
    *Example:* You see step-by-step actions in the console.
"""
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tool = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tool, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
def main():
    # Runs the agent to search for 3 AI job postings on LinkedIn in Toronto using LangChain and summarizes them.
    # Example: The agent fetches job listings and provides a summary of each.
    result = agent_executor.invoke(input={"input":"Search a 3 AI job posting on LinkedIn using langchain in Toronto Area and summarize it."})
    print(result)

if __name__ == "__main__":
    main()
