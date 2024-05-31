import re

import torch
import txtai
# from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models.ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

from Workflows import Log
from Workflows.Loaders.PDF_Loaders import LoadPDF

logger = Log("Ollama_Tasks").log()


class OllamaLocal:

    def __init__(self, model="llama2"):
        self.model = model
        logger.info(f"Selected Model: {self.model}")
        logger.info("Torch CUDA installed: " + str(torch.cuda.is_available()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selected: " + str(self.device))

    def MakeOllamaPipline(self):
        self.llama = Ollama(model=self.model)

    # "max_length": 100
    # You can add you own templates here !!!
    # OR you CAN PASS custom template here as sysPrompt
    def makeReactAgent(self,
                       sysPrompt="""Answer the following as best you can. You g=have access to the following tools:
                       {tools}
                       Use the following format:
                       
                       Question: the input question you must answer
                       Thought: you should always think about what to do
                       Action: The action to take, should be may be one of the [{tool_names}] or you think the answer step by step
                       Observation: the result of the action
                       ... (this Thought/Action/Action Input/Observatiopn can repete N times)
                       Thought: I know the final answer
                       Final Answer: the final answer to the original question
                       Parse Output: Beautify the output string
                       
                       Begin!
                       
                       Question: {input}
                       Thought: {agent_scratchpad}
                       """):
        prompt = PromptTemplate.from_template(template=sysPrompt)
        self.MakeOllamaPipline()
        ReactAgent = create_react_agent(llm=self.llama, prompt=prompt, tools=[])
        agentExec = AgentExecutor(agent=ReactAgent, tools=[], verbose=True, handle_parsing_errors=True)
        # max_iterations=20, max_execution_time=30
        return agentExec

    def RAGAgentExecuter(self, WhatIsInFile: str, MoreDescription: str):
        raise NotImplemented
        # loader = PyPDFLoader(file)
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # texts = text_splitter.split_documents(documents)
        raise NotImplemented
        name = filedialog.askdirectory(title="Load your embeddings")
        embeddings = txtai.Embeddings(path="sentence-transformers/nli-mpnet-base-v2")
        db = embeddings.load(name)
        retriever = db.as_retriever()
        tool = create_retriever_tool(
            retriever,
            WhatIsInFile,
            MoreDescription,
        )
        ReactAgent = self.makeReactAgent()
        AgentExecutor(agent=ReactAgent, tools=[tool], verbose=True, handle_parsing_errors=True)

    def RAGAgentChatExecuter(self):
        ReactAgent = self.makeReactAgent()
        AgentExecutor(agent=ReactAgent, tools=[], verbose=True, handle_parsing_errors=True)

    def promptInputParser(self, sysPrompt):
        """
          This function extracts the values between curly brackets in a string.

          Args:
            string: The string to extract the values from.

          Returns:
            A list of the values between curly brackets in the string.
          """
        Match = re.compile(r"{(\w*)}")
        return Match.findall(string=sysPrompt)

    def RAGQA_Chat(self, Query):
        result = LoadPDF().SearchEmbeddings(Query=Query)
        print(result)
        self.Chat(sysPrompt=result + "\n{Question}:")

    def Chat(self, sysPrompt: str, model="llama2"):
        """
        Chat chain from Langchain
        :param sysPrompt: One time prompt applicable as context in the chat
        :param prompt: THe formatted strings represented in system prompt
        :param model: The name of the LLM model llam2 default
        :return: None
        """
        llm = ChatOllama(model=model)
        Chatprompt = ChatPromptTemplate.from_template(sysPrompt)
        Chatchain = Chatprompt | llm | StrOutputParser()
        InputList = self.promptInputParser(sysPrompt)
        while True:
            dict = {}
            for io in InputList:
                dict[str(io)] = input(f"Give come context for:\n{io}: ")
            print(Chatchain.invoke(dict))
