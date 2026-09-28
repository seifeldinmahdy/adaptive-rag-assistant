# Adaptive RAG routing and grading

An adaptive RAG assistant first routes a question to a local knowledge base or to web
search. Local retrieval is appropriate when the question matches the knowledge base's
subject matter. Current events and unrelated subjects require web search.

A document grader checks whether retrieved passages help answer the actual question;
mere keyword overlap is insufficient. Discard irrelevant passages. If no relevant
passages remain, search the web. If relevant passages remain, generate a grounded
answer, then check both support and whether the question was answered sufficiently.

An unsupported or incomplete answer triggers web search and regeneration. Keep useful
existing evidence, add new web evidence, and pass checker feedback to the generator.
Track retry_count in state and stop once max_retries is reached. A bounded graph should
report that the answer could not be verified instead of presenting an unsupported
draft as verified. max_retries counts extra attempts after the first generation.
