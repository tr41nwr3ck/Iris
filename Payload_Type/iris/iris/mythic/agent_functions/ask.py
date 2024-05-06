from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from .helpers.tools.GetCallbackByUUIDTool import GetCallbackByUUIDTool
from .helpers.tools.GraphQLAPIWrapper import GraphQLAPIWrapper
from .helpers.tools.ExecuteGraphQLQueryTool import ExecuteGraphQLQueryTool
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from gql.transport.requests import RequestsHTTPTransport
from gql import Client, gql
from langchain import hub
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


GRAPHQL_API_KEY = "GRAPHQL_API_KEY"

class AskArguments(TaskArguments):
    def __init__(self, command_line, **kwargs):
        super().__init__(command_line, **kwargs)
        self.args = [
            CommandParameter(
                name="question",
                type=ParameterType.String,
                description="Question to prompt the LLM",
            ),
        ]

    async def parse_arguments(self):
        if len(self.command_line) == 0:
            raise Exception("Usage: {}".format(AskCommand.help_cmd))
        if self.command_line[0] == "{":
            self.load_args_from_json_string(self.command_line)
        else:
            if self.command_line[0] == '"' and self.command_line[-1] == '"':
                self.command_line = self.command_line[1:-1]
            elif self.command_line[0] == "'" and self.command_line[-1] == "'":
                self.command_line = self.command_line[1:-1]
            self.add_arg("question", self.command_line)


class AskCommand(CommandBase):
    cmd = "ask"
    needs_admin = False
    help_cmd = "ask <question>"
    description = "Ask the LLM a question about the current operation"
    version = 1
    author = "@checkymander"
    argument_class = AskArguments
    attackmapping = []
    attributes = CommandAttributes()
    chat_history = ChatMessageHistory()

    def get_message_history(self, session_id: str) -> ChatMessageHistory:
        return self.chat_history

    async def create_go_tasking(self, taskData: PTTaskMessageAllData) -> PTTaskCreateTaskingMessageResponse:
        response = PTTaskCreateTaskingMessageResponse(
            TaskID=taskData.Task.ID,
            Success=True,
        )
        self.chat_history = ChatMessageHistory(session_id=taskData.Task.ID)

        llama = Ollama(
            temperature=0,
            verbose=True,
            #model='llama3',
            model="mistral",
            base_url= "https://xbbwlp7h-11434.use.devtunnels.ms",
            #base_url= "http://localhost:11434"
        )

        question = taskData.args.get_arg("question")

        # initialize conversational memory
        # Probably going to have to change this to a file backed memory
        #memory = ChatMessageHistory(session_id="chat_history")
        self.chat_history.add_user_message(question)
        #memory.add_user_message(taskData.args.get_arg("question"))

        react_prompt = hub.pull("hwchase17/react")

        # transport = RequestsHTTPTransport(url=API_URL, headers=HEADERS, verify=False)
        # gql_client = Client(transport=transport, fetch_schema_from_transport=True)
        # gql_function = gql

        # # create an instance of GraphQLAPIWrapper
        # graphql_wrapper = GraphQLAPIWrapper(custom_headers=HEADERS, graphql_endpoint=API_URL, gql_client=gql_client, gql_function=gql_function)
        # executeGraphqlQueryTool = ExecuteGraphQLQueryTool(graphql_wrapper=graphql_wrapper)
        getCallbackByUUIDTool = GetCallbackByUUIDTool()
        tools_list = [getCallbackByUUIDTool]
        # initialize agent with tools
        agent = create_react_agent(
            tools=tools_list,
            llm=llama,
            prompt=react_prompt,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=True,
            max_iterations=5,
        )

        agent_with_chat_history = RunnableWithMessageHistory(
            agent_executor,
            self.get_message_history,
            input_messages_key="input",
            output_messages_key="output",
            history_messages_key="chat_history",
        )

        chat_response = await agent_with_chat_history.with_config(configurable={'session_id': taskData.Task.ID}).ainvoke(
            {"input": question}
        )

        await SendMythicRPCResponseCreate(MythicRPCResponseCreateMessage(
            TaskID=taskData.Task.ID,
            Response=chat_response["output"]
        ))

        response.Success = True
        print("[+] Done.")
        await SendMythicRPCTaskUpdate(MythicRPCTaskUpdateMessage(
            TaskID=taskData.Task.ID,
            UpdateCompleted = True,
            UpdateStatus = "completed"
        ))
        return response

    async def process_response(self, task: PTTaskMessageAllData, response: any) -> PTTaskProcessResponseMessageResponse:
        resp = PTTaskProcessResponseMessageResponse(TaskID=task.Task.ID, Success=True)
        return resp
