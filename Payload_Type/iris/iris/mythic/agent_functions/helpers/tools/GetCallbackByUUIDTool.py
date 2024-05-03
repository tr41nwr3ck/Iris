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
            return json.dumps(response.Results[0])
        else:
            return json.dumps({"message","Callback Not Found"})

    async def _arun(self, agent_callback_id: str):
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id)
        response = await SendMythicRPCCallbackSearch(search_message)

        if response.Success:
            print(str(response.Results[0]))
            return json.dumps(response.Results[0].__dict__)
        else:
            return json.dumps({"message","Callback Not Found"})