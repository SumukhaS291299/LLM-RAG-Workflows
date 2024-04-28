from Workflows.Tasks import Task

RAE = Task('text-generation').makeAgent()
print(RAE)
print(RAE.invoke({"question": "Write a simple sentence with whe word Blue"}))
