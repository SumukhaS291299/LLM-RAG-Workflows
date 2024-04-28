import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import SleepTool

from Workflows import Log

logger = Log("Load_Embeddings").log()


class Task:

    def __init__(self, task: str, model=""):
        self.task = task
        if not (model == ""):
            self.model = model
        else:
            logger.info(f"Selecting a model for the task {self.task}")
            self.SelectModel()
            logger.info(f"Selected Model: {self.model}")
        logger.info("Torch CUDA installed: " + str(torch.cuda.is_available()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selected: " + str(self.device))

    def SelectModel(self):
        self.task_modelMap = {"text-generation": ["distilbert/distilgpt2", "openai-community/gpt2"],
                              "text2text-generation": ["google/flan-t5-small"],
                              'summarization': ["google/flan-t5-small"]}
        self.model = self.task_modelMap.get(self.task)[0]

    def MakeTaskPipline(self):
        self.HFPipline = HuggingFacePipeline.from_model_id(task=self.task, model_id=self.model,
                                                           device_map=str(self.device),
                                                           pipeline_kwargs={"max_new_tokens": 1000})

    # "max_length": 100
    def makeAgent(self,
                  sysPrompt="Answer the following {question}, if unsure reply I don't know the answer, optional args {tool_names},{agent_scratchpad},{tools}"):
        prompt = PromptTemplate.from_template(template=sysPrompt)
        self.MakeTaskPipline()
        ReactAgent = create_react_agent(llm=self.HFPipline, prompt=prompt, tools=[])
        agentExec = AgentExecutor(agent=ReactAgent, tools=[], verbose=True, handle_parsing_errors=True)
        # max_iterations=20, max_execution_time=30
        return agentExec
