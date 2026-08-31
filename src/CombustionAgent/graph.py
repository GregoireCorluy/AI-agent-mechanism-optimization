from .state import AgentState
from langgraph.graph import StateGraph, START, END

class AgentGraph:

    def __init__(self, agent):

        self.agent = agent

        self.graph = StateGraph(AgentState)

        self.graph.add_node("router", self.router_node)
        self.graph.add_node("chat", self.chat_node)
        self.graph.add_node("retrieve", self.retrieve_node)
        #self.graph.add_node("verify", self.verify_node)
        self.graph.add_node("update", self.update_node)
        self.graph.add_node("fill", self.fill_node)

        self.graph.add_edge(START, "router")
        self.graph.add_conditional_edges(
                                    "router",
                                    self.route_after_router,
                                    {
                                        "CHAT": "chat",
                                        "RETRIEVE": "retrieve",
                                        "UPDATE": "update",
                                        "END": END,
                                    }
                                )
        self.graph.add_edge("chat", END)
        self.graph.add_edge("fill", "chat")
        self.graph.add_edge("update", "chat")
        self.graph.add_conditional_edges(
                                    "retrieve",
                                    self.route_after_retrieve,
                                    {
                                        "chat": "chat",
                                        "fill": "fill",
                                    }
                                )

        self.app = self.graph.compile()

    def router_node(self, state: AgentState):

        possible_actions = ["CHAT"]

        if all(value is not None for value in state["input_parameters"].model_dump().values()):
            possible_actions.append("END")
            possible_actions.append("UPDATE")
        elif all(value is None for value in state["input_parameters"].model_dump().values()):
            possible_actions.append("RETRIEVE")
        else:
            possible_actions.append("UPDATE") #should theoretically not exist since all the fields should be filled in after the retrieve

        selected_route = self.agent.LLM_router.define_route(state["response"], state["user_message"], state["input_parameters"], possible_actions)

        print(f"Selected route: {selected_route}")

        return {"route": selected_route}

    def chat_node(self, state: AgentState):

        message = f"""
                    CURRENT USER MESSAGE:
                    {state["user_message"]}

                    PROCESS HISTORY:
                    {state["process_history"]}

                    CURRENT INPUT PARAMETERS:
                    {state["input_parameters"]}

                    TASK:
                    Respond naturally and coherently to the user's latest message.
                    Take into account the process history, which summarizes what the agent has done since the last user's message, and the current input
                    parameters.

                    If further information is required, ask the user for it.
                    Do not invent information.
                    """

        response = self.agent.LLM_conversation.generate(message)

        return {
            "response": response
        }

    def retrieve_node(self, state: AgentState):

        LLM_retrieval_reply, input_parameters = self.agent.LLM_retrieval.retrieve_information(state["user_message"])

        print(f"\nAgent: {LLM_retrieval_reply}")
        print(f"Input parameters: {input_parameters}")

        LLM_verification_reply = self.agent.LLM_verification.verify_information(state["user_message"], input_parameters)

        print(f"\nVerification by the agent: {LLM_verification_reply}")

        LLM_update_reply, input_parameters_updated = self.agent.LLM_update.update_information(LLM_verification_reply, input_parameters)
        
        print(f"\nUpdate by the agent: {LLM_update_reply}")

        if all(value is not None for value in input_parameters_updated.model_dump().values()):
            history_entry = (
                            "RETRIEVAL RESULT: All required input parameters are currently filled. "
                            "The agent should present the extracted parameters to the user and "
                            "ask for confirmation."
                        )
        elif all(value is None for value in input_parameters_updated.model_dump().values()):
            history_entry = (
                            "RETRIEVAL RESULT: None of the input parameters have been retrieved from the user's message, all parameters will be inferred by the fill in function."
                        )
        else:
            history_entry = (
                            "RETRIEVAL RESULT: Some input parameters have been retrieved from the user's message, the other ones will be retrieved by the fill in function."
                        )

        return {"input_parameters": input_parameters_updated,
                "process_history": state["process_history"] + [history_entry]}

    # def verify_node(self, state: AgentState):
    #     result = self.agent.verify(...)
    #     return {...}

    def update_node(self, state: AgentState):

        LLM_reply, input_parameters_filled = self.agent.LLM_update.update_information(state["user_message"], state["input_parameters"])
        
        print(f"\nAgent: {LLM_reply}")

        history_entry = (
                        "UPDATE RESULT: The input parameters have been updated according to the user's request. The agent should present the extracted parameters to the user and ask for confirmation."
                    )

        return {"input_parameters": input_parameters_filled,
                "process_history": state["process_history"] + [history_entry]}

    def fill_node(self, state: AgentState):

        LLM_fill_reply, input_parameters_filled = self.agent.LLM_fill.fill_missing_information(state["input_parameters"])
        
        print(f"\nAgent: {LLM_fill_reply}")

        history_entry = (
                        "FILL RESULT: The input parameters, that were missing from the user's message, have been filled based on the context provided by the user and combined with RAG retrieval on a combustion database. The agent should present the extracted parameters to the user and ask for confirmation."
                    )

        return {"input_parameters": input_parameters_filled,
                "process_history": state["process_history"] + [history_entry]}
    
    def route_after_router(self, state: AgentState):
        return state["route"]

    def route_after_retrieve(self, state: AgentState):
        if all(value is not None for value in state["input_parameters"].model_dump().values()):
            return "chat" #maybe don't use the chat in that case but directly print it? How to format the string in the history for the LLM
        
        return "fill"