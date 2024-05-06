from langchain.tools import BaseTool
from math import pi
from typing import Union
  

class GetCallbackByUUIDTool(BaseTool):
    name = "Get Callback by UUID"
    description = "use this tool when you need to find a specific callback and know its agent_callback_id (UUID)"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")