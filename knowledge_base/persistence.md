# Checkpointing and bounded loops

A LangGraph checkpointer saves state snapshots as a graph executes. A thread_id
identifies one execution thread. InMemorySaver keeps checkpoints only while the
Python process lives; durable applications need a persistent checkpointer such as
SQLite or PostgreSQL. Checkpoints are different from automatic conversational memory:
a follow-up-aware application must explicitly preserve and use conversation history.

A loop needs an explicit stop condition. In this assistant a failed answer check
causes another web search only while retry_count is below max_retries. With a budget
of two retries, at most three answers are generated. LangGraph's recursion_limit is
an additional guard, not the main business rule used to terminate the retry loop.

Source: https://docs.langchain.com/oss/python/langgraph/persistence (teaching notes).
