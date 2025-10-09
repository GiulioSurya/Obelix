# Blackboard Pipeline Framework

## Project Overview

This project extends the existing Obelix multi-provider LLM agent framework to support a hybrid blackboard-pipeline architecture. The goal is to create independent, modular components ("building blocks") that can be composed to create both linear pipelines and intelligent manager-agent coordination systems.

## Core Concepts

### Blackboard Architecture
The blackboard serves as a universal memory center that:
- Stores all agent execution results  
- Tracks pipeline execution state and history  
- Facilitates communication between master and agents through task queues  
- Provides complete visibility for debugging and monitoring  
- Acts as the single source of truth for the entire system  

### Independent Components Philosophy
Every component must be completely independent and self-contained:
- **Agents**: Execute tasks and write results to the blackboard. The agent’s `execute` method writes the assistant’s response to the blackboard.  
- **Pipelines**: Orchestrate agent sequences (sequential, parallel, merge, loop, others?).  
- **Manager**: An agent that can register other agents (whose schema it acquires through `get_agent_schema`) or tools (whose schema it acquires through `create_schema`). It can orchestrate agents entirely through the blackboard, deciding which agent or tool to call. It can also coordinate post-pipeline feedback and intelligent restarts, typically triggered by human feedback.  
- **Blackboard**: Central memory accessible by all components. It is the core of the framework. It can register a pipeline of agents, assign a master that calls the agents, or combine both — creating a pipeline and defining an entry point for the master that coordinates execution from that moment onward.  

## Requirements

### 1. Blackboard Component
- **Universal Memory**: Store all agent results.  
- **Execution Tracking**: Track pipeline states, current steps, completed steps, etc.  
- **Task Queue**: Enable master-to-agent communication through task assignment. When the manager decides to call an agent, it creates a task (a class to be defined with specific attributes). The agent receives and executes it. If a pipeline exists, the task is executed by the agent, and the pipeline resumes from that point until completion.  
- **Debug Support**: Complete execution history for debugging and analysis.  
- **Data Persistence**: Results accessible across different pipeline executions (still under evaluation, not an essential requirement).  

### 2. Independent Agent Component
- **Self-Contained**: Each agent operates independently.  
- **Blackboard Integration**: Automatic result writing to the blackboard.  
- **Task Processing**: Ability to receive and execute tasks from the master gathered from the blackboard.  
- **State Awareness**: Knowledge of execution context and step names. This is already implemented in the chat history, which must be accessible from the blackboard.  

### 3. Simple Pipeline Component
- **Fluent API**: Chain agents using `.then()`, `.parallel()`, `.merge()`, etc.  
- **Step Naming**: Each pipeline step has a unique identifier.  
- **State Tracking**: Current and completed steps written to the blackboard.  
- **Restart Capability**: Ability to restart from any specific step.  
- **Input Propagation**: Results flow between agents in the chain.  

### 4. Independent Master Component
- **Agent Registry**: Register and manage available agents.  
- **Intelligent Analysis**: Analyze blackboard state and user feedback.  
- **Decision Making**: Determine which agents to call or which steps to restart.  
- **Task Coordination**: Create and assign tasks to specific agents.  
- **Feedback Loop**: Handle user feedback and coordinate appropriate responses.  

## Architecture Principles

### Modularity
- Each component can be used independently.  
- Components can be combined in different configurations.  
- No tight coupling between components (except through the blackboard).  

### Transparency
- All operations are recorded in the blackboard.  
- Complete execution history available for analysis.  
- State is always queryable and inspectable.  

### Intelligent Restart
- The master can analyze execution history.  
- Determine optimal restart points based on needed changes.  
- Avoid redundant re-execution of unaffected steps.  

### Composability
- Mix and match agents in different pipeline configurations.  
- Reuse agents across multiple pipelines.  
- Combine sequential and parallel execution patterns.  

## Success Criteria

### Functional Requirements
- Agents execute independently and write to the blackboard.  
- Pipelines orchestrate agent sequences effectively.  
- Master analyzes feedback and coordinates intelligent responses.  
- System restarts from optimal points without redundant work.  
- All components function as independent building blocks.  

### Non-Functional Requirements
- Complete operational transparency through the blackboard.  
- Easy debugging via execution history.  
- Flexible composition of agents and pipelines.  
- Minimal coupling between components.  
- Clear separation of concerns.  

## Example Usage Scenario

Consider a Text-to-SQL pipeline with the following agents:
1. **Table Selector**: Identifies relevant database tables.  
2. **Column Pruner**: Selects necessary columns.  
3. **Query Enhancer**: Adds filters and conditions.  
4. **SQL Generator**: Creates the final SQL query.  

### Initial Execution
User provides: “Show me sales data for last month.”  
The pipeline executes all steps sequentially, results stored in the blackboard.  

### User Feedback
User requests: “Add customer name column to the result.”  

### Master Coordination
1. The master analyzes the blackboard state.  
2. Detects that Table Selector and Column Pruner have already run.  
3. Determines that Column Pruner excluded the customer name column.  
4. Decides to restart from the Query Enhancer step.  
5. Query Enhancer and SQL Generator re-execute with new requirements.  
6. Final result includes the customer name column.  

### Benefits Demonstrated
- No redundant table selection.  
- Intelligent restart point selection.  
- Complete execution transparency.  
- Efficient feedback handling.  

## Example Usage Code

> This description is purely illustrative and not binding for the project. Any improvement that enhances the framework can be freely implemented.

```python
# Setup Independent Building Blocks
blackboard = Blackboard()

# Create agents
table_selector = BaseAgent(
    system_message="system message",
    agent_name="table selector agent",
    description="Agent capable of reading database tables and selecting relevant columns for the query"
)

columns_pruner = BaseAgent(
    system_message="system message",
    agent_name="column pruner agent",
    description="Agent capable of reading database tables and selecting relevant columns for the query"
)

# and other agents...

# Create manager
manager_agent = Manager(system_message="system_message")
manager_agent.register_agents([table_selector, columns_pruner])

# Create Linear Pipeline (Building Block)
text2sql_pipeline = (
    Pipeline(name="text2sql")
    .then(table_selector, "table_selection")
    .then(column_pruner, "column_pruning")
    .then(query_enhancer, "query_enhancement")
    .then(sql_generator, "sql_generation")
)

# Create blackboard
blackboard = Blackboard(pipeline=text2sql_pipeline, manager=manager_agent)

blackboard.set_manager_entry_point(after="sql_generation")

# === INITIAL EXECUTION ===
user_query = "Show me sales data for last month"
result = text2sql_pipeline.execute(user_query)

print(f"Generated SQL: {result}")
print(f"Pipeline State: {blackboard.pipeline_state}")
print(f"Completed Steps: {blackboard.pipeline_state['text2sql']['completed_steps']}")

# === USER FEEDBACK SCENARIO ===
user_feedback = "Add customer name column to the result"

# Master analyzes and decides
decision = master.analyze_and_decide(user_feedback)
print(f"Master Decision: {decision}")

# Master executes decision (intelligent restart)
final_result = master.execute_decision(decision)
print(f"Updated SQL: {final_result}")

# === DEBUGGING CAPABILITIES ===
print("Execution History:")
for entry in blackboard.execution_history:
    print(f"  {entry['timestamp']}: {entry['agent']} - {entry['action']}")

# Agent-specific results
table_result = blackboard.read_agent_result("table_selector")
print(f"Table Selector Result: {table_result}")

column_result = blackboard.read_agent_result("column_pruner")
print(f"Column Pruner Result: {column_result}")

# === ALTERNATIVE USAGE PATTERNS ===

# 1. Parallel Execution Example
parallel_pipeline = (
    SimplePipeline("parallel_analysis", blackboard)
    .then(table_selector, "table_selection")
    .parallel(
        (column_pruner, "column_analysis"),
        (query_enhancer, "query_analysis")
    )
    .then(sql_generator, "final_generation")
)

# 2. Single Agent Usage (Independent Building Block)
single_result = table_selector.execute("Find tables related to customer data")
print(f"Single Agent Result: {single_result}")

# 3. Master Task Assignment
master.add_task("enhance_query", query_enhancer.name, {
    "input": "Add date filters for last month",
    "step": "manual_enhancement"
})

# Agent checks for tasks
task_executed = query_enhancer.check_for_tasks()
if task_executed:
    print("Task completed by agent")

# 4. Pipeline Restart from Specific Step
modified_result = text2sql_pipeline.restart_from_step(
    "table_selection",
    "Find tables related to product sales and customer info"
)

# === COMPOSABILITY DEMONSTRATION ===

# Reuse agents in a different pipeline
report_pipeline = (
    SimplePipeline("report_generation", blackboard)
    .then(table_selector, "report_table_selection")
    .then(sql_generator, "report_sql_generation")
)

# Mix sequential and parallel patterns
complex_pipeline = (
    SimplePipeline("complex_analysis", blackboard)
    .then(table_selector, "initial_selection")
    .parallel(
        (column_pruner, "column_optimization"),
        (query_enhancer, "query_optimization")
    )
    .then(sql_generator, "optimized_generation")
    .then(table_selector, "validation_check")  # Reuse same agent
)

# === MONITORING AND DEBUG OUTPUT ===
print("\n=== BLACKBOARD STATE ===")
print(f"All Agent Results: {blackboard.agent_outputs}")
print(f"Pipeline States: {blackboard.pipeline_state}")
print(f"Task Queue: {blackboard.tasks}")
print(f"Full History: {blackboard.execution_history}")
