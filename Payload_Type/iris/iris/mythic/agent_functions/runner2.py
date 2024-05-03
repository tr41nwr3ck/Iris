from .helpers.tools.GraphQLAPIWrapper import GraphQLAPIWrapper
from .helpers.tools.ExecuteGraphQLQueryTool import ExecuteGraphQLQueryTool
from langchain_community.chat_models import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from gql.transport.requests import RequestsHTTPTransport
from gql import Client, gql
from langchain import hub

graphql_token = ""
API_URL = "https://10.30.26.108:7443/v1/graphql"

llama = ChatOllama(
    temperature=0.3,
    model='llama3',
    base_url= "http://localhost:11434"
)

# initialize conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=ChatMessageHistory())

HEADERS = {
    "content-type":"application/json",
    "x-hasura-admin-secret":graphql_token
}


react_prompt = hub.pull("hwchase17/react")
transport = RequestsHTTPTransport(url=API_URL, headers=HEADERS, verify=False)
gql_client = Client(transport=transport, fetch_schema_from_transport=True)
gql_function = gql

# create an instance of GraphQLAPIWrapper
graphql_wrapper = GraphQLAPIWrapper(custom_headers=HEADERS, graphql_endpoint=API_URL, gql_client=gql_client, gql_function=gql_function)
executeGraphqlQueryTool = ExecuteGraphQLQueryTool(graphql_wrapper=graphql_wrapper)
tools_list = [executeGraphqlQueryTool]

# initialize agent with tools
agent = create_react_agent(
    tools=tools_list,
    llm=llama,
    prompt=react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools_list)

agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools_list, verbose=True, memory=memory)

agent_chain.invoke(
    {"input": "Run this query: query GetCallback {callback(where: {agent_callback_id: {_eq: \"a934b822-a3cd-4e34-844b-a715e1170c8a\"}}) {agent_callback_id}}"},
    config={"configurable": {"session_id": "chat_history"}},
)

# agent.agent.llm_chain.prompt =s new_prompt
# agent('Can you please return the hostname of the callback with the callback id of "a934b822-a3cd-4e34-844b-a715e1170c8a"')
# #TODO get working based on https://colab.research.google.com/drive/1XxqokxiUZYxilnq4mWbPd5SGG-Yhy-py?usp=sharing#scrollTo=S2fuQNCGeCAQ