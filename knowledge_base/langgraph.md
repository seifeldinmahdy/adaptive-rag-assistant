# LangGraph state, nodes, and edges

LangGraph models an application as a graph of nodes connected by edges. A shared State
stores values such as the question, documents, answer, and retry count. A StateGraph
can use a TypedDict as its state schema. Nodes are Python functions that read state
and return partial updates. Fields normally overwrite previous values; an annotated
reducer can define how concurrent or accumulated updates combine.

Normal edges always transfer control from one node to the next. Conditional edges
use a routing function to select the next node from the current state. START is the
entry point and END terminates execution. Compile the graph before invoking it.
Cycles let the system revise an answer instead of following only a linear pipeline.

Source: https://docs.langchain.com/oss/python/langgraph/graph-api (teaching notes).
