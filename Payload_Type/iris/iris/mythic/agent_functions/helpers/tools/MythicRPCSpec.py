from llama_index.core.tools.tool_spec.base import BaseToolSpec
from mythic_container.MythicCommandBase import *
from mythic_container.MythicRPC import *
import json

class MythicRPCSpec(BaseToolSpec):
    spec_functions = ["get_callback_by_uuid_async", "task_callback"]
    _scope:str = None

    def __init__(self, scope: str):
        self._scope: str = scope

    async def get_callback_by_uuid_async(self, agent_callback_id: str) -> str:
        """Finds a specific callback by its agent_callback_id (UUID)"""
        search_message = MythicRPCCallbackSearchMessage(AgentCallbackUUID=self._scope,
                                                        SearchCallbackUUID=agent_callback_id)
        response = await SendMythicRPCCallbackSearch(search_message)

        if response.Success:
            return response.Results[0].to_json()
        else:
            return json.dumps({"message":"Callback Not Found"})
    
    async def task_callback(self, agent_callback_id:str, command:str, params: str):
        """Executes a command on a callback specified by its agent_callback_id with parameters specified by a json string of parameter names and parameter values"""
        response = await SendMythicRPCTaskCreate(MythicRPCTaskCreateMessage(AgentCallbackID=agent_callback_id, CommandName=command, Params=params))

        if response.Success:
            return "Task issued."
        else:
            return f"Failed to issue task: {response.Error}"
