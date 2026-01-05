"""
Sidekick Tools Module

This module provides a collection of tools that can be used by a LangGraph agent (Sidekick).
The tools enable the agent to interact with the web, manage files, execute Python code,
search the internet, query Wikipedia, and send push notifications.

Required environment variables (in .env file):
- PUSHOVER_TOKEN: API token for Pushover notification service
- PUSHOVER_USER: User key for Pushover notification service
- SERPER_API_KEY: API key for Serper (Google Search API)
"""

# Browser automation for web interaction
from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

# Environment variable management
from dotenv import load_dotenv
import os

# HTTP requests for push notifications
import requests

# Core LangChain tool framework
from langchain.agents import Tool

# File system management tools
from langchain_community.agent_toolkits import FileManagementToolkit

# Wikipedia query tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Python code execution tool (REPL = Read-Eval-Print Loop)
from langchain_experimental.tools import PythonREPLTool

# Google Search API wrapper
from langchain_community.utilities import GoogleSerperAPIWrapper


# ============================================================================
# Configuration: Load environment variables and initialize API wrappers
# ============================================================================

# Load environment variables from .env file (overrides any existing env vars)
load_dotenv(override=True)

# Pushover configuration for sending push notifications to mobile devices
# Get credentials from: https://pushover.net/
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

# Initialize Google Search API wrapper (requires SERPER_API_KEY in .env)
# Serper provides a simple API for Google Search results
serper = GoogleSerperAPIWrapper()


# ============================================================================
# Browser Tools: Web automation using Playwright
# ============================================================================

async def playwright_tools():
    """
    Creates and returns Playwright browser tools for web automation.
    
    This function:
    1. Starts a Playwright instance (handles browser automation)
    2. Launches a Chromium browser in headful mode (visible window)
    3. Creates a toolkit with browser automation tools (navigate, click, extract text, etc.)
    4. Returns the tools along with browser and playwright objects (for cleanup later)
    
    Returns:
        tuple: (list of tools, browser object, playwright object)
               - tools: List of LangChain tools for browser automation
               - browser: Browser instance (keep reference for cleanup)
               - playwright: Playwright instance (keep reference for cleanup)
    
    Note: This is an async function, so it must be called with 'await' in async contexts.
          The browser runs in headful mode (headless=False) so you can see it working.
          
    Example usage:
        tools, browser, playwright = await playwright_tools()
        # ... use tools ...
        await browser.close()  # Clean up when done
        await playwright.stop()
    """
    # Start Playwright - this manages the browser automation engine
    playwright = await async_playwright().start()
    
    # Launch a Chromium browser window (headless=False means you can see it)
    browser = await playwright.chromium.launch(headless=False)
    
    # Create a toolkit that provides LangChain tools for browser automation
    # These tools allow the agent to: navigate to URLs, click elements, fill forms,
    # extract text, take screenshots, etc.
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    
    # Return tools list along with browser/playwright objects (needed for cleanup)
    return toolkit.get_tools(), browser, playwright


# ============================================================================
# Push Notification Tool
# ============================================================================

def push(text: str):
    """
    Send a push notification to the user's mobile device via Pushover.
    
    Args:
        text (str): The message text to send in the notification
    
    Returns:
        str: "success" if the notification was sent
    
    Note: Requires PUSHOVER_TOKEN and PUSHOVER_USER to be set in .env file.
          The user must have the Pushover app installed and configured.
    """
    # Send POST request to Pushover API with notification data
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


# ============================================================================
# File Management Tools
# ============================================================================

def get_file_tools():
    """
    Returns file management tools for the agent to read/write files.
    
    The FileManagementToolkit provides tools for:
    - Reading files from the filesystem
    - Writing files to the filesystem
    - Listing directory contents
    - Moving/copying files
    
    All operations are restricted to the "sandbox" directory for safety.
    
    Returns:
        list: List of LangChain tools for file operations
    """
    # Create toolkit with operations restricted to "sandbox" directory
    # This provides a safe workspace for file operations
    toolkit = FileManagementToolkit(root_dir="sandbox")
    return toolkit.get_tools()


# ============================================================================
# Other Tools: Search, Wikipedia, Python REPL, File Management, Push
# ============================================================================

async def other_tools():
    """
    Assembles and returns a collection of tools for the Sidekick agent.
    
    This function creates and combines multiple tool types:
    1. Push notification tool - send alerts to user's phone
    2. File management tools - read/write files in sandbox directory
    3. Web search tool - search Google via Serper API
    4. Wikipedia tool - query Wikipedia for information
    5. Python REPL tool - execute Python code dynamically
    
    Returns:
        list: Combined list of all tools that the agent can use
    
    Note: This is an async function for consistency with playwright_tools(),
          even though most of these tools don't require async operations.
    
    Example usage:
        tools = await other_tools()
        # tools is now a list that can be passed to an agent
    """
    # 1. Create push notification tool
    # Wraps the push() function as a LangChain Tool that the agent can call
    push_tool = Tool(
        name="send_push_notification", 
        func=push, 
        description="Use this tool when you want to send a push notification"
    )
    
    # 2. Get file management tools (read, write, list files in sandbox/)
    file_tools = get_file_tools()

    # 3. Create web search tool using Google Serper API
    # The serper.run function executes Google searches and returns results
    tool_search = Tool(
        name="search",
        func=serper.run,  # serper.run(query) performs the actual search
        description="Use this tool when you want to get the results of an online web search"
    )

    # 4. Create Wikipedia query tool
    # First create the API wrapper, then wrap it in a Tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # 5. Create Python REPL (Read-Eval-Print Loop) tool
    # Allows the agent to execute arbitrary Python code - use with caution!
    # This is powerful but can be dangerous if not properly sandboxed
    python_repl = PythonREPLTool()
    
    # Combine all tools into a single list
    # file_tools is already a list, so we concatenate it with the other tools
    combined = file_tools + [push_tool, tool_search, python_repl, wiki_tool]
    return combined


# ============================================================================
# Tool metadata helpers
# ============================================================================

def extract_tool_metadata(tool, source: str = None) -> dict:
    """
    Extract human-friendly metadata from a LangChain Tool-like object.

    Returns a dict with keys: name, display_name, description, source
    """
    name = getattr(tool, "name", None) or (getattr(tool, "tool_name", None) if hasattr(tool, "tool_name") else None)
    if not name:
        # Fallback to function name if available
        func = getattr(tool, "func", None)
        if func and hasattr(func, "__name__"):
            name = func.__name__
        else:
            name = str(tool)

    display_name = name.replace("_", " ").title()

    # Prefer explicit description attribute, then function docstring
    description = getattr(tool, "description", None)
    if not description:
        func = getattr(tool, "func", None)
        if func and func.__doc__:
            description = func.__doc__.strip().split("\n")[0]

    return {
        "name": name,
        "display_name": display_name,
        "description": description or "",
        "source": source or getattr(tool, "__module__", "unknown"),
    }


def build_tool_metadata_map(tools, source: str = None) -> dict:
    """
    Build a mapping of tool.name -> metadata dict for a list of tools.
    """
    metadata_map = {}
    for t in tools:
        meta = extract_tool_metadata(t, source=source)
        metadata_map[meta["name"]] = meta
    return metadata_map

