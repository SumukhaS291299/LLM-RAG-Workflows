from Workflows.Tasks import Task

# RAE = Task('text-generation').makeAgent()
# print(RAE)
# print(RAE.invoke({"question": "Write a simple sentence with whe word Blue"}))

from Workflows.Ollama_Tasks import OllamaLocal

# RAE = OllamaLocal().makeReactAgent()
# print(RAE.invoke({"question": "Write a simple sentence with whe word Blue"}))

Chat = OllamaLocal().Chat("Answer the following questions {QUESTIONS} based on {CONTEXT}")
