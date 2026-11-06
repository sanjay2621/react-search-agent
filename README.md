# react-search-agent

A ReAct-style search agent built with LangChain and Tavily.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your keys
```

Keys needed: `OPENAI_API_KEY` and `TAVILY_API_KEY`. LangSmith tracing is
optional.

## Run

```bash
uv run react-search-agent
```
