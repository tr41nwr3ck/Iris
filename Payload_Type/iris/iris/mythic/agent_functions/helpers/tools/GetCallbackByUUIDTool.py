from langchain.tools import BaseTool
from typing import Type, Union
from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
from pydantic import BaseModel, Field
import asyncio
import json

  
class GetCallbackByUUIDSchema(BaseModel):
    agent_callback_id: str = Field(description="Should be a UUID")
    
class GetCallbackByUUIDTool(BaseTool):
    name = "Get Callback by UUID"
    description = "use this tool when you need to find a specific callback and know its agent_callback_id (UUID)"
    args_schema: Type[GetCallbackByUUIDSchema] = GetCallbackByUUIDSchema
    
    def _run(self, agent_callback_id: str):
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id)
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(SendMythicRPCCallbackSearch(search_message))

        if response.Success:
            return json.dumps(response.Results[0])
        else:
            return json.dumps({"message","Callback Not Found"})

    def _arun(self, agent_callback_id: str):
        raise NotImplementedError("This tool does not support async")