from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import set_global_tokenizer
from huggingface_hub import hf_hub_download
#from llama_index.readers.graphql import GraphQLReader
from .data.irisgraphql import IrisGraphQLReader
import torch
import os
from .data.mythic_query import query_all_data


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

    async def query_all(self, token):
        uri = "https://127.0.0.1:7443/v1/graphql"
        headers = {
            "content-type":"application/json",
            "x-hasura-admin-secret":token
        }

        reader = IrisGraphQLReader(uri = uri, headers = headers)
        
        # Get stuff from database
        await query_all_data(reader)
        return reader.documents

    async def generate_text(self, llm_model_path, embeddings, n_gpu_layers, taskData):
        print("[+] Querying Data.")
        #index = VectorStoreIndex.from_documents(self.query_graphql(taskData.Secrets[GRAPHQL_API_KEY]), embed_model=embeddings)
        index = VectorStoreIndex.from_documents(await self.query_all(taskData.Secrets[GRAPHQL_API_KEY]), embed_model=embeddings)
        #index = VectorStoreIndex.from_documents(self.query_files(graphql_key), embed_model=embeddings)
        prompt_template = """
### System:
{system_message}
### User:
{prompt}
### Assistant:
    """
        print("[+] Loading Model.")
        await SendMythicRPCTaskUpdate(MythicRPCTaskUpdateMessage(
            TaskID=taskData.Task.ID,
            UpdateStatus = "Generating response"
        ))
        llm = LlamaCPP(
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=llm_model_path,
            temperature=0.1,
            max_new_tokens=256,
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=3900,
            # kwargs to pass to __call__()
            generate_kwargs={},
            # kwargs to pass to __init__()
            # set to at least 1 to use GPU
            model_kwargs={"n_gpu_layers": n_gpu_layers},
            verbose=False,    
        )
        print("[+] Creating Chat Memory.")
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        prompt_template += f"""Question: {taskData.args.get_arg("question")}"""

        ### Todo: Update this prompt to follow the right template based on LLM specified
        # Prompt template can be found in the tokenizer_config.json file
        single_turn_prompt = f"<｜begin▁of▁sentence｜><|Assistant|>\nYou are an AI Assistant who can answer technical questions based on information.\n<|User|>\n{prompt_template}\n<|Assistant|><｜end▁of▁sentence｜>\n"     
        print("[+] Starting Chat Engine.")   
        chat_engine = index.as_chat_engine(
            chat_mode="context", #https://github.com/run-llama/llama_index/blob/2ba13544cd2583418cbeade5bea45ff1da7bb7b8/llama-index-core/llama_index/core/chat_engine/types.py#L298
            memory=memory,
            system_prompt=(
                single_turn_prompt
            ),
            query_engine = index.as_query_engine(llm=llm),
            llm=llm,
        )
        print(f"[+] Sending Prompt: {single_turn_prompt}")
        chat_response = chat_engine.chat(single_turn_prompt)
        return chat_response.response
    def get_embeddings(self, embedding_model):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        return embeddings
    def get_reranker(self, reranking_model, device):
        rerank_tokenizer = AutoTokenizer.from_pretrained(reranking_model)
        rerank_model = AutoModelForSequenceClassification.from_pretrained(reranking_model).to(device)
        return (rerank_tokenizer, rerank_model)    
    def query_files(self, token):
        return SimpleDirectoryReader(
        "/Mythic/iris/mythic/agent_functions/test_data"
        ).load_data()

    async def create_go_tasking(self, taskData: PTTaskMessageAllData) -> PTTaskCreateTaskingMessageResponse:
        response = PTTaskCreateTaskingMessageResponse(
            TaskID=taskData.Task.ID,
            Success=True,
        )

        # Map the model name with its appropraite gguf file
        model_map = {
            "TheBloke/neural-chat-7B-v3-3-GGUF": "neural-chat-7b-v3-3.Q5_K_M.gguf",
            "bartowski/WhiteRabbitNeo-7B-v1.5a-GGUF": "WhiteRabbitNeo-7B-v1.5a-Q6_K.gguf"
        }

        # Sometimes configs are based on other models and don't have the config.json for the auto tokenizer, so we can map the proper ones here
        config_map = {
            "TheBloke/neural-chat-7B-v3-3-GGUF" : "TheBloke/neural-chat-7B-v3-3-GGUF",
            "bartowski/WhiteRabbitNeo-7B-v1.5a-GGUF": "WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a",
            "BAAI/bge-reranker-base": "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3"
        }

        for buildParam in taskData.BuildParameters:
            if buildParam.Name == "LLM":
                llm_model = buildParam.Value
            if buildParam.Name == "Embedding":
                embedding_model = buildParam.Value
            if buildParam.Name == "Reranker":
                reranker_model = buildParam.Value

        set_global_tokenizer(
            # Token indices sequence length is longer than the specified maximum sequence length for this model (576 > 512). Running this sequence through the model will result in indexing errors
            # This is caused by the embedding_model not matching the reranker_model
            AutoTokenizer.from_pretrained(
                config_map[reranker_model],
                model_max_length=512
            ).encode
        )
        
        if GRAPHQL_API_KEY not in taskData.Secrets:
            raise Exception("Missing GraphQL API Key")

        if torch.cuda.is_available():
        # traditional Nvidia cuda GPUs
            device = torch.device("cuda:0")
            n_gpu_layers = 10
        elif torch.backends.mps.is_available():
            # for macOS M1/M2s
            device = torch.device("mps")
            n_gpu_layers = 10
        else:
            device = torch.device("cpu")
            n_gpu_layers = 0

        try:
            llm_model_path = hf_hub_download(llm_model, filename=model_map[llm_model], local_files_only=True)
        except Exception as e:
            print(e)
            llm_model_path = hf_hub_download(llm_model, filename=model_map[llm_model])

        try:
            embeddings = self.get_embeddings(embedding_model)
        except Exception as e:
            print(e)
            raise Exception("Failed to get embedding model")

        try:
            (rerank_tokenizer, rerank_model) = self.get_reranker(reranker_model, device)
        except Exception as e:
            print(e)
            raise Exception("Failed to get reranker")

        print("[+] All models downloaded.")
        await SendMythicRPCTaskUpdate(MythicRPCTaskUpdateMessage(
            TaskID=taskData.Task.ID,
            UpdateStatus = "Querying database"
        ))

        chat_response = await self.generate_text(llm_model_path, embeddings, n_gpu_layers, taskData)
        await SendMythicRPCResponseCreate(MythicRPCResponseCreateMessage(
            TaskID=taskData.Task.ID,
            Response=chat_response,
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
