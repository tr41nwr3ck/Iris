from langchain.tools import BaseTool
import json
from pydantic import BaseModel, Field
from typing import Type, Optional
import GraphQLAPIWrapper

class GetCallbacksToolSchema(BaseModel):
    variables: Optional[dict] = Field(description="Optional dictionary of variables for the query")

class GetCallbacksTool(BaseTool):
    graphql_wrapper: GraphQLAPIWrapper
    # Maybe make tools with predefined queries that the LLM can call? E.g. GetTaskOutputQueryTool and GetCallbackName agent tools
    name = "GetAllCallbacksTool"
    description = """\
        Input to this tool is a detailed and correct GraphQL query along with any variables as a dictionary, output is a result from making a request to the API.
        If the query is not correct, an error message will be returned.
        If an error is returned with 'Bad request' in it, rewrite the query and try again.
        If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.
        The payloadtype name should not include "iris" as that's the name of the code this agent is running under

        Example Input: query GetCallbacks{{callback(where:{{_not:{{payload:{{payloadtype:{{_not:{{name:{{_neq:"iris"}}}}}}}}}}}}){{active agent_callback_id architecture description domain external_ip host id init_callback integrity_level ip last_checkin os pid process_name timestamp user}}}}
        Example Input: query GetCallback {{callback(where: {{agent_callback_id: {{_eq: "a934b822-a3cd-4e34-844b-a715e1170c8a"}}}}) {{active agent_callback_id architecture description domain external_ip host id init_callback integrity_level ip last_checkin os pid process_name timestamp user}}}}

        use this tool when you need to execute a GraphQL query against an external API
        given a query as a string and an optional variables dictionary object.

        To use the tool, you must provide at least the 'query' parameter:
        ['query']

        If providing variables (optional), you must provide a query and dictionary of variables:
        ['query', 'variables']
    """
    args_schema: Type[GetCallbacksToolSchema] = GetCallbacksToolSchema

    def _run(self, query: str, variables: Optional[dict] = {}) -> str:
        """
        Execute a GraphQL query and return the response.

        :param query: A string containing a GraphQL query.
        :param variables: An Optional dictionary containing the variables for the query.
        :return: The json response from the GraphQL API.
        """

        result = self.graphql_wrapper.run(query, variables)
        return json.dumps(result, indent=2)


    def _arun(self, query: str, variables: dict = {}) -> str:
        raise NotImplementedError("This tool does not support async")