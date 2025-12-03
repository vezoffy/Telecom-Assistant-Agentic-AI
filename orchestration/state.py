from typing import TypedDict, List, Any, Annotated
import operator

class AgentState(TypedDict):
    query: str
    category: str
    response: str
    history: Annotated[List[Any], operator.add]
    customer_id: str
