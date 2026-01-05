"""
Sidekick Agent - Self-Evaluating AI Assistant

This module implements a self-evaluating AI agent using LangGraph that:
1. Works on tasks using various tools (browser, search, Python REPL, files, etc.)
2. Evaluates its own work against success criteria
3. Iteratively improves based on feedback
4. Asks for user input when clarification is needed

The agent uses a worker-evaluator loop:
- Worker: Uses LLM with tools to complete tasks
- Evaluator: Uses LLM to assess if success criteria is met
- Routing: Determines next step (continue work, ask user, or finish)

Architecture:
    START -> worker -> (tools <-> worker)* -> evaluator -> (worker | END)
    
The graph loops until either:
- Success criteria is met (task completed successfully)
- User input is needed (agent has question or is stuck)
"""

# LangGraph core components for building state machines
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # Helper to merge message lists

# Environment variable management
from dotenv import load_dotenv

# Pre-built tool execution node for LangGraph
from langgraph.prebuilt import ToolNode

# LLM and memory components
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # In-memory conversation history

# Message types for chat interface
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Type hints
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field

# Tool functions from sidekick_tools module
from sidekick_tools import playwright_tools, other_tools

# Utilities
import uuid  # Generate unique IDs for conversation threads
import asyncio  # For async operations
from datetime import datetime  # For timestamps in prompts
import re  # For stripping HTML when producing Markdown-safe content

# Load environment variables (API keys, etc.)
load_dotenv(override=True)


# ============================================================================
# State Definition: The data structure passed between graph nodes
# ============================================================================

class State(TypedDict):
    """
    State structure for the LangGraph agent.
    
    This defines what data flows through the graph at each step.
    TypedDict provides type hints while remaining compatible with dict operations.
    """
    # Conversation messages (user prompts, assistant responses, tool results)
    # Annotated with add_messages so LangGraph automatically merges new messages
    messages: Annotated[List[Any], add_messages]
    
    # What the agent is trying to achieve (e.g., "Summarize this article in 3 bullet points")
    success_criteria: str
    
    # Feedback from evaluator if previous attempt didn't meet criteria
    # None on first attempt, contains feedback string on subsequent attempts
    feedback_on_work: Optional[str]
    
    # Whether the evaluator determined the task is complete
    success_criteria_met: bool
    
    # Whether the agent needs user input (question, clarification, or stuck)
    user_input_needed: bool


# ============================================================================
# Evaluator Output Schema: Structured output from evaluator LLM
# ============================================================================

class EvaluatorOutput(BaseModel):
    """
    Structured output format for the evaluator LLM.
    
    Pydantic BaseModel ensures the LLM returns data in this exact structure.
    The evaluator LLM is configured to return this format using with_structured_output().
    """
    # Detailed feedback on the assistant's work (what was good, what needs improvement)
    feedback: str = Field(description="Feedback on the assistant's response")
    
    # True if the task has been completed successfully according to success_criteria
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    
    # True if the assistant needs user input (asked a question, needs clarification, or is stuck)
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


# ============================================================================
# Sidekick Class: Main agent implementation
# ============================================================================

class Sidekick:
    """
    Self-evaluating AI assistant that works on tasks using tools and evaluates its own work.
    
    The Sidekick uses a worker-evaluator pattern:
    - Worker: LLM with tools that attempts to complete tasks
    - Evaluator: LLM that assesses if the work meets success criteria
    - Graph: Routes between worker, tools, and evaluator until done
    
    Usage:
        sidekick = Sidekick()
        await sidekick.setup()  # Initialize tools, LLMs, and graph
        history = await sidekick.run_superstep("Your task", "Success criteria", [])
        sidekick.cleanup()  # Close browser and cleanup resources
    """
    
    def __init__(self):
        """
        Initialize Sidekick instance.
        
        Creates instance variables that will be set during setup().
        Each Sidekick instance has a unique ID for conversation thread management.
        """
        # LLM instances (initialized in setup())
        self.worker_llm_with_tools = None  # LLM for worker node (has access to tools)
        self.evaluator_llm_with_output = None  # LLM for evaluator (returns structured output)
        self.tools = None  # List of all tools available to the worker
        self.llm_with_tools = None  # (Unused - can be removed)
        
        # LangGraph instance (compiled in build_graph())
        self.graph = None
        
        # Unique ID for this Sidekick instance (used as thread_id for conversation history)
        self.sidekick_id = str(uuid.uuid4())
        
        # Memory/checkpointer for storing conversation history
        self.memory = MemorySaver()
        
        # Tracking: tools used and current status (for UI feedback)
        self.tools_used: List[str] = []
        self.status: str = "Idle"
        
        # Browser automation objects (for cleanup)
        self.browser = None
        self.playwright = None

    async def setup(self):
        """
        Initialize the Sidekick: load tools, configure LLMs, and build the graph.
        
        This must be called before using the Sidekick. It:
        1. Loads browser tools (Playwright) and other tools (search, Python, files, etc.)
        2. Configures the worker LLM with tool bindings
        3. Configures the evaluator LLM with structured output
        4. Builds and compiles the LangGraph state machine
        
        Note: This is async because browser initialization is asynchronous.
        
        Example:
            sidekick = Sidekick()
            await sidekick.setup()
            # Now ready to use
        """
        # 1. Initialize browser tools (Playwright for web automation)
        # Returns: (tools_list, browser_instance, playwright_instance)
        self.tools, self.browser, self.playwright = await playwright_tools()
        
        # 2. Add additional tools (search, Wikipedia, Python REPL, file management, push notifications)
        self.tools += await other_tools()

        # Build a metadata map for available tools so we can display friendly info in the UI
        try:
            from sidekick_tools import build_tool_metadata_map
            self.tool_metadata_map = build_tool_metadata_map(self.tools)
        except Exception:
            self.tool_metadata_map = {}
        
        # 3. Create worker LLM and bind tools to it
        # bind_tools() allows the LLM to see available tools and call them when needed
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        
        # 4. Create evaluator LLM with structured output
        # with_structured_output() ensures the LLM returns EvaluatorOutput format
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        
        # 5. Build the graph (worker -> tools <-> evaluator flow)
        await self.build_graph()

    # ========================================================================
    # Worker Node: The main agent that uses tools to complete tasks
    # ========================================================================

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Worker node: LLM that uses tools to work on the task.
        
        This is the main "brain" of the agent. It:
        1. Receives the current state (conversation, success criteria, feedback)
        2. Constructs a system message with instructions and context
        3. Calls the LLM (which can decide to use tools or respond directly)
        4. Returns the LLM's response (which may contain tool calls)
        
        The worker LLM has access to all tools and can:
        - Browse the web (Playwright)
        - Search Google (Serper)
        - Query Wikipedia
        - Execute Python code
        - Read/write files
        - Send push notifications
        
        Args:
            state: Current state containing messages, success_criteria, feedback_on_work
            
        Returns:
            Dict with "messages" key containing the LLM's response
            The response may contain tool_calls if the LLM wants to use tools
        """
        # Build system message with instructions for the worker
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    This is the success criteria:
    {state["success_criteria"]}
    You should reply either with a question for the user about this assignment, or with your final response.
    If you have a question for the user, you need to reply by clearly stating your question. An example might be:

    Question: please clarify whether you want a summary or a detailed answer

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """

        # Add feedback if this is a retry after evaluation failure
        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Ensure system message is in the message list (update existing or add new)
        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        # Add system message at the beginning if not found
        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Set status to thinking (helps UI show progress)
        self.status = "Thinking..."

        # Invoke the LLM with tools (LLM can choose to call tools or respond directly)
        response = self.worker_llm_with_tools.invoke(messages)

        # If the response contains tool calls, record which tools were requested
        if hasattr(response, "tool_calls") and response.tool_calls:
            self.status = "Using tools..."
            try:
                for tc in response.tool_calls:
                    # Tool call objects may have different shapes; try common fields
                    name = getattr(tc, "name", None) or getattr(tc, "tool_name", None) or str(tc)
                    # Build enriched metadata from registry when available
                    meta = getattr(self, "tool_metadata_map", {}).get(name)
                    entry = {
                        "name": name,
                        "display_name": meta["display_name"] if meta else name.replace("_", " ").title(),
                        "description": meta.get("description", "") if meta else "",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "source": meta.get("source") if meta else None,
                    }
                    # Avoid duplicates by tool name
                    if not any(e.get("name") == name for e in self.tools_used):
                        self.tools_used.append(entry)
            except Exception:
                # Non-fatal: if extraction fails, just ignore
                pass

        # Return updated state with the LLM's response
        # If response contains tool_calls, worker_router will route to "tools" node
        # Otherwise, it will route to "evaluator" node
        return {
            "messages": [response],
        }

    def worker_router(self, state: State) -> str:
        """
        Router function: Determines next step after worker node.
        
        Checks if the worker's last message contains tool calls.
        If yes -> route to "tools" node to execute the tools
        If no -> route to "evaluator" node to evaluate the response
        
        Args:
            state: Current state with messages
            
        Returns:
            "tools" if last message has tool_calls, "evaluator" otherwise
        """
        last_message = state["messages"][-1]

        # Check if the LLM wants to use tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"  # Execute tools first
        else:
            return "evaluator"  # Evaluate the response directly

    def format_conversation(self, messages: List[Any]) -> str:
        """
        Format conversation messages into a readable string.
        
        Converts the message list into a simple text format for the evaluator.
        Only includes user and assistant messages (excludes system messages and tool messages).
        
        Args:
            messages: List of message objects
            
        Returns:
            Formatted string with conversation history
        """
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                # Use placeholder if message has no text (e.g., only tool calls)
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    # ========================================================================
    # Evaluator Node: Assesses if the worker's output meets success criteria
    # ========================================================================

    def evaluator(self, state: State) -> State:
        """
        Evaluator node: LLM that assesses if the worker's work meets success criteria.
        
        This node acts as a quality control check. It:
        1. Reviews the entire conversation and the worker's final response
        2. Compares the response against the success criteria
        3. Determines if the task is complete or if more work is needed
        4. Determines if user input is needed (question asked, clarification needed, stuck)
        5. Provides feedback for improvement if criteria not met
        
        The evaluator uses structured output (EvaluatorOutput) to ensure consistent format.
        
        Args:
            state: Current state with messages, success_criteria, feedback_on_work
            
        Returns:
            Updated state with:
            - feedback_on_work: Evaluator's feedback
            - success_criteria_met: Whether task is complete
            - user_input_needed: Whether user interaction is required
            - messages: Includes evaluator's feedback message
        """
        last_response = state["messages"][-1].content

        # System prompt for evaluator role
        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""

        # User prompt with context for evaluation
        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state["messages"])}

    The success criteria for this assignment is:
    {state["success_criteria"]}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
        # Add context if this is a retry after previous feedback
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        # Prepare messages for evaluator LLM
        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        # Set status so UI can show we're evaluating
        self.status = "Evaluating response..."

        # Invoke evaluator LLM (returns structured EvaluatorOutput)
        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        
        # Prepare evaluator feedback as Markdown-safe text (no raw HTML)
        feedback_text = re.sub(r"<[^>]+>", "", eval_result.feedback).strip()
        timestamp = datetime.utcnow().isoformat() + "Z"
        eval_md = f"🧠 **Evaluation — {timestamp}**\n\n> {feedback_text}"

        # Update state with evaluation results (plain text / markdown)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": eval_md,
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

        # Back to idle after evaluation (unless the graph continues with tools)
        self.status = "Idle"
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        """
        Router function: Determines next step after evaluator node.
        
        Checks evaluation results to decide:
        - If success_criteria_met OR user_input_needed -> END (task complete or needs user)
        - Otherwise -> worker (continue working, try again with feedback)
        
        Args:
            state: Current state with evaluation results
            
        Returns:
            "END" if done or needs user input, "worker" if should continue
        """
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"  # Task complete or user interaction needed
        else:
            return "worker"  # Continue working with feedback

    # ========================================================================
    # Graph Building: Constructs the LangGraph state machine
    # ========================================================================

    async def build_graph(self):
        """
        Build and compile the LangGraph state machine.
        
        Creates a graph with this flow:
        
        START
          ↓
        worker (uses LLM with tools)
          ↓
        [worker_router checks for tool_calls]
          ├─→ tools (execute tool calls) ──┐
          │                                  │ (loop back to worker)
          └─→ evaluator (assess work) ───────┘
                    ↓
            [route_based_on_evaluation]
                    ├─→ END (done or needs user)
                    └─→ worker (continue with feedback)
        
        The graph loops until:
        - Success criteria is met
        - User input is needed (question asked, stuck, etc.)
        
        Uses MemorySaver checkpointer to maintain conversation history.
        """
        # Initialize graph builder with State structure
        graph_builder = StateGraph(State)

        # Add nodes (functions that process state)
        graph_builder.add_node("worker", self.worker)  # Main agent that uses tools
        graph_builder.add_node("tools", ToolNode(tools=self.tools))  # Execute tool calls
        graph_builder.add_node("evaluator", self.evaluator)  # Assess if work is complete

        # Add edges (routing between nodes)
        
        # Conditional edge: worker -> (tools OR evaluator)
        # worker_router checks if last message has tool_calls
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        
        # Unconditional edge: tools -> worker (always loop back after tool execution)
        graph_builder.add_edge("tools", "worker")
        
        # Conditional edge: evaluator -> (worker OR END)
        # route_based_on_evaluation checks if task is complete or needs user
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END}
        )
        
        # Entry point: START -> worker
        graph_builder.add_edge(START, "worker")

        # Compile the graph with memory/checkpointer for conversation history
        # checkpointer stores state between invocations (for multi-turn conversations)
        self.graph = graph_builder.compile(checkpointer=self.memory)

    # ========================================================================
    # Execution: Run the agent on a task
    # ========================================================================

    async def run_superstep(self, message, success_criteria, history):
        """
        Execute the agent on a task and return the conversation history.
        
        This is the main entry point for running the Sidekick. It:
        1. Creates initial state with the user's message and success criteria
        2. Invokes the graph (runs worker -> tools <-> evaluator loop)
        3. Returns formatted conversation history including response and feedback
        
        The graph will loop until:
        - Success criteria is met (task complete)
        - User input is needed (agent has question or is stuck)
        
        Args:
            message: User's input message (string or list of messages)
            success_criteria: Criteria for successful task completion (string)
            history: Previous conversation history (list of message dicts)
            
        Returns:
            Updated history list with:
            - User message
            - Assistant's final response (result["messages"][-2])
            - Evaluator feedback (result["messages"][-1])
            
        Example:
            history = await sidekick.run_superstep(
                "Summarize this article",
                "Summary should be 3 bullet points",
                []
            )
        """
        # Configuration for graph execution
        # thread_id identifies this conversation thread in memory/checkpointer
        config = {"configurable": {"thread_id": self.sidekick_id}}

        # Reset per-run tracking (tools used etc.) so UI shows only items for this run
        self.tools_used = []
        self.status = "Thinking..."

        # Create initial state
        state = {
            "messages": message,  # User's input (can be string or list)
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,  # No feedback on first attempt
            "success_criteria_met": False,  # Not met yet
            "user_input_needed": False,  # Not needed yet
        }
        
        # Run the graph (worker -> tools <-> evaluator loop until done)
        result = await self.graph.ainvoke(state, config=config)
        
        # Ensure final status is Idle and expose tools used (helpful for UI)
        final_status = getattr(self, "status", "Idle")
        final_tools = getattr(self, "tools_used", [])
        self.status = "Idle"

        # Format results for return
        # result["messages"][-2] is the worker's final response
        # result["messages"][-1] is the evaluator's feedback message
        # Produce Markdown-safe messages (strip any HTML tags returned by tools/LLMs)
        user_text = re.sub(r"<[^>]+>", "", str(message)).strip()
        user_md = f"**👤 User**\n\n{user_text}"

        assistant_content = result["messages"][-2].content
        # Strip any HTML tags from assistant content to avoid unsafe HTML in chat
        assistant_text = re.sub(r"<[^>]+>", "", str(assistant_content)).strip()
        assistant_md = f"**🤖 Sidekick**\n\n{assistant_text}"

        feedback_content = result["messages"][-1].content
        feedback_text = re.sub(r"<[^>]+>", "", str(feedback_content)).strip()
        timestamp = datetime.utcnow().isoformat() + "Z"
        feedback_md = f"🧠 **Evaluation — {timestamp}**\n\n> {feedback_text}"

        user = {"role": "user", "content": user_md}
        reply = {"role": "assistant", "content": assistant_md}
        feedback = {"role": "assistant", "content": feedback_md}

        # Attach status/tools info to the returned history structure by returning a tuple
        return history + [user, reply, feedback], {"status": final_status, "tools_used": final_tools}

    # ========================================================================
    # Cleanup: Release resources (browser, playwright)
    # ========================================================================

    def cleanup(self):
        """
        Clean up resources: close browser and stop Playwright.
        
        This should be called when done using the Sidekick to free resources.
        Handles both cases:
        - If async event loop is running: schedules cleanup as tasks
        - If no event loop: runs cleanup directly
        
        Note: In some cases (e.g., notebook environments), cleanup may not
        complete if the event loop is already closed. This is usually fine.
        
        Example:
            await sidekick.setup()
            # ... use sidekick ...
            sidekick.cleanup()
        """
        if self.browser:
            try:
                # Try to get running event loop (for async environments)
                loop = asyncio.get_running_loop()
                # Schedule cleanup tasks
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # No event loop running, use asyncio.run() directly
                # This handles synchronous contexts
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
