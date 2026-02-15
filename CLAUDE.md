# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BaYeAgents is a research agent system built with the `deepagents` framework. It creates AI agents with internet search capabilities via Tavily API to conduct research and generate reports.

## Development Environment

**Package Manager**: This project uses `uv` (not pip) for dependency management.

**Python Version**: 3.14+ (specified in `.python-version` and `pyproject.toml`)

### Running the Project

```bash
# Install dependencies (if needed)
uv sync

# Run the main application
python main.py
```

### Installing Dependencies

When adding new dependencies, update `pyproject.toml` and run:
```bash
uv lock
uv sync
```

## Architecture

The project is currently a single-file application ([main.py](main.py)) structured around:

1. **`internet_search()` function**: A tool function wrapping TavilyClient for web searches
   - Parameters: `query`, `max_results` (default 5), `topic` ("general"|"news"|"finance"), `include_raw_content`

2. **`create_deep_agent()`**: From the deepagents library, creates the agent with:
   - `tools`: List of tool functions the agent can use
   - `system_prompt`: Instructions that guide agent behavior

3. **Agent invocation**: `agent.invoke()` takes a messages dict with format:
   ```python
   {"messages": [{"role": "user", "content": "your query here"}]}
   ```

4. **Response format**: The result contains a `messages` list; the last message has the agent's response in `.content`

## Environment Variables

Required in `.env`:
- `ANTHROPIC_API_KEY`: For Anthropic models via LangChain
- `TAVILY_API_KEY`: For Tavily web search API

## Key Dependencies

- `deepagents>=0.3.8`: Agent framework (built on LangChain)
- `tavily-python>=0.7.19`: Web search API client
- `langchain`: Chat model integration via `init_chat_model`

## Notes

- Currently has no test framework or tests
- The README.md is a placeholder
- All code is currently in main.py - consider modularizing as the project grows
