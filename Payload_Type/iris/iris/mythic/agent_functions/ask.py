from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from .helpers.tools.GetCallbackByUUIDTool import GetCallbackByUUIDTool
from .helpers.tools.GraphQLAPIWrapper import GraphQLAPIWrapper
from .helpers.tools.ExecuteGraphQLQueryTool import ExecuteGraphQLQueryTool
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from gql.transport.requests import RequestsHTTPTransport
from gql import Client, gql
from langchain import hub


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
    attributes = CommandAttributes(
    )
    async def create_go_tasking(self, taskData: PTTaskMessageAllData) -> PTTaskCreateTaskingMessageResponse:
        response = PTTaskCreateTaskingMessageResponse(
            TaskID=taskData.Task.ID,
            Success=True,
        )
        llama = ChatOllama(
            temperature=0,
            model='llama3',
            base_url= "https://xbbwlp7h-11434.use.devtunnels.ms"
            #base_url= "http://localhost:11434"
        )

        # initialize conversational memory
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=ChatMessageHistory())

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

        agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                         tools=tools_list,
                                                         verbose=True, 
                                                         memory=memory)
        agent_chain.max_iterations = 1

        question = taskData.args.get_arg("question")

        chat_response = await agent_chain.ainvoke(
            {"input": question},
            config={"configurable": {"session_id": "chat_history"}},
        )


        await SendMythicRPCResponseCreate(MythicRPCResponseCreateMessage(
            TaskID=taskData.Task.ID,
            Response=str(chat_response),
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
