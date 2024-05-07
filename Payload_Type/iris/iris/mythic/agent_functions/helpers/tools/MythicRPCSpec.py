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
    spec_functions = ["get_callback_by_uuid_async", "task_callback"]

    async def get_callback_by_uuid_async(self, agent_callback_id: str) -> str:
        """Finds a specific callback by its agent_callback_id (UUID)"""
        print(f"\nAgent Callback ID: {agent_callback_id}")
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=agent_callback_id,
                                                        SearchCallbackUUID=agent_callback_id)
        response = await SendMythicRPCCallbackSearch(search_message)

        if response.Success:
            return response.Results[0].to_json()
        else:
            return json.dumps({"message":"Callback Not Found"})
    
    async def task_callback(self, agent_callback_id:str, command:str, params: str):
        """Executes a command on a callback specified by its agent_callback_id with parameters specified by a json string of parameter names and parameter values"""
        print(f"Command to execute: {agent_callback_id}")
        print(f"Parameters: {params}")
        print(f"Agent ID: {agent_callback_id}")

        response = await SendMythicRPCTaskCreate(MythicRPCTaskCreateMessage(AgentCallbackID=agent_callback_id, CommandName=command, Params=params))

        if response.Success:
            return "Task issued."
        else:
            return f"Failed to issue task: {response.Error}"
