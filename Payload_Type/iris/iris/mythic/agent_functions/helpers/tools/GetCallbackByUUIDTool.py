from langchain.tools import BaseTool
from typing import Dict, Self, Type, Union
from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from pydantic import BaseModel, Field, ValidationError, root_validator
import asyncio
import json

  
class GetCallbackByUUIDSchema(BaseModel):
    agent_callback_id: str = Field(description="Should be a UUID")
    
class GetCallbackByUUIDTool(BaseTool):
    name = "Get Callback by UUID"
    description = "use this tool when you need to find a specific callback and know its agent_callback_id (UUID)"
    args_schema: Type[GetCallbackByUUIDSchema] = GetCallbackByUUIDSchema
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        return values

    def _run(self, agent_callback_id: str):
        loop = asyncio.get_running_loop()
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id)
        response = loop.run_until_complete(SendMythicRPCCallbackSearch(search_message))
        #response = await SendMythicRPCCallbackSearch(search_message)
        if response.Success:
            agent : MythicRPCCallbackSearchMessageResult = response.Results[0]
            response = f"""
{agent.AgentCallbackID=}
{agent.Description=}
{agent.User=}
{agent.Host=}
{agent.PID=}
{agent.Ip=}
{agent.ProcessName=}
{agent.IntegrityLevel=}
{agent.CryptoType=}
{agent.Os=}
{agent.Architecture=}
{agent.Domain=}
            """


            return response
        else:
            return "Callback Not Found"

    async def _arun(self, agent_callback_id: str):
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id)
        response = await SendMythicRPCCallbackSearch(search_message)

        if response.Success:
            agent : MythicRPCCallbackSearchMessageResult = response.Results[0]
            response = f"""
{agent.AgentCallbackID=}
{agent.Description=}
{agent.User=}
{agent.Host=}
{agent.PID=}
{agent.Ip=}
{agent.ProcessName=}
{agent.IntegrityLevel=}
{agent.CryptoType=}
{agent.Os=}
{agent.Architecture=}
{agent.Domain=}
            """
            return response
        else:
            return "Callback Not Found"