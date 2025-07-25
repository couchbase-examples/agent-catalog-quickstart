from agentc.catalog import tool
from pydantic import BaseModel


# Although Python uses duck-typing, the specification of models greatly improves the response quality of LLMs.
# It is highly recommended that all tools specify the models of their bound functions using Pydantic or dataclasses.
# class SalesModel(BaseModel):
#     input_sources: list[str]
#     sales_formula: str


# Only functions decorated with "tool" will be indexed.
# All other functions / module members will be ignored by the indexer.
@tool
def temp(<<< Replace me with your input type! >>>) -> <<< Replace me with your output type! >>>:
    """temp"""

    <<< Replace me with your Python code! >>>
