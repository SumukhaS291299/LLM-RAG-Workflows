from Workflows.Tasks import Task

# RAE = Task('text-generation').makeAgent()
# print(RAE)
# print(RAE.invoke({"question": "Write a simple sentence with whe word Blue"}))

from Workflows.Ollama_Tasks import OllamaLocal

RAE = OllamaLocal(model="gemma:2b").makeReactAgent()
# print(RAE.invoke({"input": "Write a simple sentence with whe word Blue"}))
print(RAE.invoke({"input": input("Ask me anything:\n>")}))

# Chat = OllamaLocal().Chat("Given {CONTEXT} Answer the following questions {QUESTIONS}")
# Chat = OllamaLocal().RAGQA_Chat(Query="Best traffic control system")
