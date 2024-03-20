from mythic_container.PayloadBuilder import *
from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
from llama_index.core import set_global_tokenizer
import torch


class Iris(PayloadType):
    name = "iris"
    file_extension = ""
    author = "@checkymander"
    supported_os = [
        SupportedOS("iris")
    ]
    wrapper = False
    wrapped_payloads = []
    note = """
    This payload allows you to ask questions about your current operation
    """
    supports_dynamic_loading = False
    mythic_encrypts = True
    translation_container = None
    agent_type = "service"
    agent_path = pathlib.Path(".") / "iris" / "mythic"
    agent_code_path = pathlib.Path(".") / "iris"  / "agent_code"
    agent_icon_path = agent_path / "agent_functions" / "iris.svg"
    build_steps = [
        BuildStep(step_name="Download Models", step_description="Downloading selected models"),
        BuildStep(step_name="Start Agent", step_description="Starting agent callback"),
    ]
    build_parameters = [
        BuildParameter(
            name="LLM",
            parameter_type=BuildParameterType.ChooseOne,
            choices=["TheBloke/neural-chat-7B-v3-3-GGUF"],
            default_value="TheBloke/neural-chat-7B-v3-3-GGUF",
            description="The base LLM model to use"
        ),
        BuildParameter(
            name="Embedding",
            parameter_type=BuildParameterType.ChooseOne,
            choices=["TaylorAI/gte-tiny"],
            default_value="TaylorAI/gte-tiny",
            description="The embedding model to use"
        ),
        BuildParameter(
            name="Reranker",
            parameter_type=BuildParameterType.ChooseOne,
            choices=["BAAI/bge-reranker-base"],
            default_value="BAAI/bge-reranker-base",
            description="The reranker model"
        ),
    ]
    c2_profiles = []

    def get_embeddings(self, embedding_model):
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        return embeddings

    def get_reranker(self, reranking_model, device):
        rerank_tokenizer = AutoTokenizer.from_pretrained(reranking_model)
        rerank_model = AutoModelForSequenceClassification.from_pretrained(reranking_model).to(device)
        return (rerank_tokenizer, rerank_model)



    async def build(self) -> BuildResponse:
        model_map = {
            "TheBloke/neural-chat-7B-v3-3-GGUF": "neural-chat-7b-v3-3.Q5_K_M.gguf",
        }

        # Check if path exists, if no download it.
        try:
            llm_model_path = hf_hub_download(self.get_parameter("LLM"), filename=model_map[self.get_parameter("LLM")], local_files_only=True)
        except:
            llm_model_path = hf_hub_download(self.get_parameter("LLM"), filename=model_map[self.get_parameter("LLM")])
        
        try:
            embeddings = self.get_embeddings(self.get_parameter("Embedding"))
        except:
            print("Failed to get embedding model.")

        try:
            if torch.cuda.is_available():
            # traditional Nvidia cuda GPUs
                device = torch.device("cuda:0")
            elif torch.backends.mps.is_available():
                # for macOS M1/M2s
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

            (rerank_tokenizer, rerank_model) = self.get_reranker(self.get_parameter("Reranker"), device)
        except:
            print("Failed to get reranker")

        # this function gets called to create an instance of your payload
        resp = BuildResponse(status=BuildStatus.Success)
        ip = "127.0.0.1"
        create_callback = await SendMythicRPCCallbackCreate(MythicRPCCallbackCreateMessage(
            PayloadUUID=self.uuid,
            C2ProfileName="",
            User="iris",
            Host="iris",
            Ip=ip,
            IntegrityLevel=3,
        ))
        if not create_callback.Success:
            logger.info(create_callback.Error)
        else:
            logger.info(create_callback.CallbackUUID)
        return resp