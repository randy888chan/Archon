# Archon Chat Process - Sequence Diagram

This diagram illustrates the temporal flow of interactions between components in the Archon Chat process, showing how the system helps users build AI agents using Pydantic AI.

## Detailed Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant UI as Streamlit UI
    participant Thread as Thread Manager
    participant Flow as Agentic Flow
    participant Reasoner as Reasoner LLM
    participant DocSystem as Documentation System
    participant Coder as Pydantic AI Coder
    participant Router as Router Agent
    participant PromptRefiner as Prompt Refiner
    participant ToolsRefiner as Tools Refiner
    participant AgentRefiner as Agent Refiner
    participant EndConvo as End Conversation Agent

    %% Initial request
    User->>UI: Describe AI agent to build
    UI->>Thread: Generate thread ID
    UI->>Flow: Send request with thread ID

    %% Scope definition
    Flow->>Reasoner: Define scope for agent
    Reasoner->>DocSystem: List documentation pages
    DocSystem-->>Reasoner: Return available pages
    Reasoner-->>Flow: Return scope document
    Flow->>Flow: Save scope to workbench/scope.md

    %% Initial agent creation
    Flow->>Coder: Create agent with scope
    Coder->>DocSystem: Retrieve relevant documentation
    DocSystem-->>Coder: Return documentation chunks
    Coder->>DocSystem: List documentation pages
    DocSystem-->>Coder: Return available pages
    Coder->>DocSystem: Get specific page content
    DocSystem-->>Coder: Return full page content
    Coder-->>Flow: Return agent code
    Flow-->>UI: Stream response to user
    UI-->>User: Display agent code

    %% User feedback loop
    User->>UI: Provide feedback or request
    UI->>Flow: Send user message
    Flow->>Router: Route user message

    %% Decision branching
    alt User wants to continue coding
        Router-->>Flow: Route to coder agent
        Flow->>Coder: Update agent with feedback
        Coder->>DocSystem: Retrieve additional documentation
        DocSystem-->>Coder: Return documentation
        Coder-->>Flow: Return updated agent code
        Flow-->>UI: Stream response
        UI-->>User: Display updated code
    else User wants to refine agent
        Router-->>Flow: Route to refinement process

        %% Parallel refinement processes
        par Refine prompt
            Flow->>PromptRefiner: Refine agent prompt
            PromptRefiner-->>Flow: Return refined prompt
        and Refine tools
            Flow->>ToolsRefiner: Refine agent tools
            ToolsRefiner->>DocSystem: Retrieve tool documentation
            DocSystem-->>ToolsRefiner: Return documentation
            ToolsRefiner-->>Flow: Return refined tools
        and Refine agent definition
            Flow->>AgentRefiner: Refine agent definition
            AgentRefiner->>DocSystem: Retrieve agent documentation
            DocSystem-->>AgentRefiner: Return documentation
            AgentRefiner-->>Flow: Return refined agent definition
        end

        Flow->>Coder: Apply refinements
        Coder-->>Flow: Return refined agent code
        Flow-->>UI: Stream response
        UI-->>User: Display refined code
    else User wants to end conversation
        Router-->>Flow: Route to end conversation
        Flow->>EndConvo: Generate final response
        EndConvo-->>Flow: Return final message
        Flow-->>UI: Stream final response
        UI-->>User: Display final agent code and instructions
    end

    %% Optional additional iterations
    Note over User,UI: Process can repeat with additional feedback
```

## Key Components and Interactions

1. **User and UI Layer**:

   - User interacts with the Streamlit UI
   - UI manages the conversation history and thread ID

2. **Workflow Orchestration**:

   - Agentic Flow (LangGraph) coordinates the entire process
   - Router Agent determines the next steps based on user messages

3. **Agent Creation Process**:

   - Reasoner LLM defines the scope and architecture
   - Pydantic AI Coder implements the agent code
   - Documentation System provides relevant information

4. **Refinement Process**:

   - Prompt Refiner improves the agent's system prompt
   - Tools Refiner enhances the agent's tools
   - Agent Refiner optimizes the agent definition

5. **Conversation Flow**:
   - Initial request → Scope definition → Agent creation
   - Feedback loop: Continue coding, Refine agent, or End conversation
   - Final response with instructions for using the agent

This sequence diagram captures the temporal flow of the Archon Chat process, showing how different components interact over time to create and refine an AI agent based on the user's requirements.
