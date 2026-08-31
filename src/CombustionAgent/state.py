from typing import TypedDict

from .parameters import InputParameters

class AgentState(TypedDict):

    # Latest message from the user
    user_message: str

    # History of all the process
    process_history: list[str] | None

    # Current extracted parameters
    # None means that no parameters have been established yet
    input_parameters: InputParameters | None

    # Decision made by the router
    route: str | None

    # Final response to show to the user
    response: str | None