# The way to do this is to generate a bunch of hypothetical questions from the FAQ, index these in the vDB
# Then for the user prompt do a two stage inference with very small CTX size which only determines 
# if the user is asking a question related to items specifically mentioned on the FAQ. Then you can retrieve the relevant FAQ section or source document 
# accordingly only if the score is within a threshold


from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.utilities import GraphQLAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from typing import Iterable


graphql_token = "CUyDhLFD5EQM8PbxuifKwWWd1nzicI"
graphql_uri = "https://10.30.26.108:7443/v1/graphql"
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

from typing import Dict, List, Optional

import yaml
from llama_index.core.readers.base import BaseReader
from langchain_core.documents import BaseDocumentTransformer, Document


class IrisGraphQLReader(BaseReader):
    """GraphQL reader.

    Combines all GraphQL results into the Document used by LlamaIndex.

    Args:
        uri (str): GraphQL uri.
        headers (Optional[Dict]): Optional http headers.

    """
    def __init__(
        self,
        uri: Optional[str] = None,
        headers: Optional[Dict] = None,
    ) -> None:
        """Initialize with parameters."""

        try:
            from gql import Client
            from gql.transport.requests import RequestsHTTPTransport

        except ImportError:
            raise ImportError("`gql` package not found, please run `pip install gql`")
        if uri:
            if uri is None:
                raise ValueError("`uri` must be provided.")
            if headers is None:
                headers = {}
            transport = RequestsHTTPTransport(url=uri, headers=headers, verify=False)
            self.client = Client(transport=transport, fetch_schema_from_transport=True)
            self.documents: List[Document] = []  # Initialize an empty list of documents

    def load_data(self, query: str, variables: Optional[Dict] = None) -> List[Document]:
        """Run query with optional variables and turn results into documents.

        Args:
            query (str): GraphQL query string.
            variables (Optional[Dict]): optional query parameters.

        Returns:
            List[Document]: A list of documents.

        """
        try:
            from gql import gql

        except ImportError:
            raise ImportError("`gql` package not found, please run `pip install gql`")
        if variables is None:
            variables = {}

        result = self.client.execute(gql(query), variable_values=variables)

        # for key in result:
        #     entry = result[key]
        #     if isinstance(entry, list):
        #         for v in entry:
        #             text_chunks = self.split_text(yaml.dump(v))
        #             self.documents.extend([Document(text=chunk) for chunk in text_chunks])
        #     else:
        #         text_chunks = self.split_text(yaml.dump(entry))
        #         self.documents.extend([Document(text=chunk) for chunk in text_chunks])
        for key in result:
            entry = result[key]
            if isinstance(entry, list):
                self.documents.extend([Document(page_content=yaml.dump(v)) for v in entry])
            else:
                self.documents.append(Document(page_content=yaml.dump(entry)))        

        return self.documents

def query_all(token):
    uri = "https://10.30.26.108:7443/v1/graphql"
    headers = {
        "content-type":"application/json",
        "x-hasura-admin-secret":token
    }

    reader = IrisGraphQLReader(uri = uri, headers = headers)
    
    # Get stuff from database
    query_all_data(reader)
    return reader.documents


# https://docs.llamaindex.ai/en/stable/examples/vector_stores/chroma_auto_retriever/
# Should add metadata to the graphql stuff pulled
def query_all_data(reader: str):
    # print("Getting Task output")
    # query_task_output(reader)
    # print("Getting payload data")
    # query_payloads(reader)
    print("Getting callback data")
    #query_callbacks(reader)
    query_callback_hosts(reader)

def query_callback_hosts(reader: IrisGraphQLReader):
    task_query = """query GetHosts{callback(where:{_not:{payload:{payloadtype:{_not:{name:{_neq:"iris"}}}}}}){active agent_callback_id host id}}"""
    query_generic(reader,task_query)

def query_task_output(reader: IrisGraphQLReader):
    task_query = """query GetTasks{task{id agent_task_id callback_id command_name completed display_params display_id is_interactive_task operator_id operation_id responses{id response_escape}parent_task_id}}"""
    query_generic(reader,task_query)

def query_payloads(reader:IrisGraphQLReader):
    query_generic(reader, """query GetPayloads{payload(where:{_not:{payloadtype:{_not:{name:{_neq:"iris"}}}}}){id callbacks{agent_callback_id architecture crypto_type description display_id domain external_ip extra_info host id init_callback integrity_level ip last_checkin os operator_id operation_id pid process_name registered_payload_id timestamp user}}}""")

def query_callbacks(reader: IrisGraphQLReader):
    query_generic(reader, """query MyQuery{callback(where:{_not:{payload:{payloadtype:{_not:{name:{_neq:"iris"}}}}}}){active agent_callback_id architecture description domain external_ip host id init_callback integrity_level ip last_checkin os pid process_name timestamp user}}""")

def query_generic(reader: IrisGraphQLReader, query: str):
    #reader.load_text(query, variables={})
    reader.load_data(query, variables={})

def split_documents(documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.text)
            metadatas.append(doc.metadata)
        return text_splitter.create_documents(texts, metadatas=metadatas)

def get_text_from_docs(documents: Iterable[Document]) -> List[str]:
    texts = []
    for doc in documents:
        texts.append(doc.text)
    return texts

def write_list_to_file(filename, my_list: Iterable[Document]):
    with open(filename, 'a') as f:
        for item in my_list:
            f.write("%s\n" % item.page_content)
def write_item_to_file(filename, page_content):
    with open(filename, 'a') as f:
            f.write("%s\n" % page_content)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



tools = load_tools(
    ["graphql"],
    graphql_endpoint="https://10.30.26.108:7443/v1/graphql",
    graphql_token = "CUyDhLFD5EQM8PbxuifKwWWd1nzicI"
)

local_llm = "llama3"

# Get our data from Mythic
docs_list = query_all(graphql_token)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=GPT4AllEmbeddings(),
)


retriever = vectorstore.as_retriever()
llm = ChatOllama(model=local_llm, 
                 format="json", 
                 temperature=0)


agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


### Generate
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = prompt | llm | JsonOutputParser()
question = "Give me a list of hosts"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)


# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a helpful assistant who helps operators answer questions about a set of data. You should be as thorough as possible with your response, hosts should be returned by hostnames only<|eot_id|><|start_header_id|>user<|end_header_id|>
#     Here is the user question: {question}\nHere is the relevant Data: {document} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["question", "document"],
# )

# LLM
llm = ChatOllama(model=local_llm, 
                 temperature=0)

# Chain
rag_chain = prompt | llm | StrOutputParser()
question = "Give me a list of hosts"
docs = retriever.invoke(question)
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

# ### Answer Grader 

# # LLM
# llm = ChatOllama(model=local_llm, format="json", temperature=0)

# # Prompt
# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
#     answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
#     useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
#      <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
#     \n ------- \n
#     {generation} 
#     \n ------- \n
#     Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["generation", "question"],
# )

# answer_grader = prompt | llm | JsonOutputParser()
# answer_grader.invoke({"question": question,"generation": generation})