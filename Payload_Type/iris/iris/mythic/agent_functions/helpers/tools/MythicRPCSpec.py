from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from langchain.tools import BaseTool
from typing import Dict, Self, Type, Union
from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from pydantic import BaseModel, Field, ValidationError, root_validator
import asyncio
import json

class MythicRPCSpec(BaseToolSpec):
    spec_functions = ["get_callback_by_uuid_async"]


    # def get_callback_by_uuid(self, agent_callback_id: str) -> str:
    #     return "Sync not supported."

    async def get_callback_by_uuid_async(self, agent_callback_id: str) -> str:
        """Finds a specific callback by its agent_callback_id (UUID)"""
        print(f"\nAgent Callback ID: {agent_callback_id}")
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id,
                                                        SearchCallbackUUID=agent_callback_id)
        response = await SendMythicRPCCallbackSearch(search_message)

        if response.Success:
    #         response_str = ""
    #         for result in response.Results:
    #             response_str += f"""
    # ==============================================
    # {result.AgentCallbackID}
    # ==============================================
    # {result.AgentCallbackID=}
    # {result.Description=}
    # {result.User=}
    # {result.Host=}
    # {result.PID=}
    # {result.Ip=}
    # {result.ProcessName=}
    # {result.IntegrityLevel=}
    # {result.CryptoType=}
    # {result.Os=}
    # {result.Architecture=}
    # {result.Domain=}
    # ==============================================
    # """
            #print(response_str)
            #return response_str
            return response.Results[0].to_json()
        else:
            return json.dumps({"message":"Callback Not Found"})