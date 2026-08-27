import torch
import sys
import json
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

# TO DO:
# - Fourth model for suggestions
# - Add temperature for models
# - chat with different iterations of the json, can receive partially filled in json as input
# - Ask at the end if the suggestion of parameters is correct
# - Improve description of Pydantic and check what the Pydantic verification is doing
# - Explore other LLMs
# - Constrain output generation for information retrieval
# - Have an LLM that iteratively specifically converts values
# - Check how to manage the history and different messages
# - let the agent think first: first generation step is analyzing the input and second generation provides the structured output; use another LLM for performing the conversion; use Python to perform the deterministic conversion and LLM just identifies the fields with the units
# - Verification also with a structured json
# - Python code to do the conversion from one unit to another, or use LLM
# - Verify again what the update has done?
# - Add a memory: notes to keep for later, keep messages of the user (?)
# - LLM router
# - convert units etc. at the end with a deterministic method
# - Ask at the end if the user agrees with the retrieved parameters
# - Create Langgraph which will choose the possibilities in function of the current state, state-dependent action
# - One LLM to take notes/memory
# - Explain to the user what is happening behind the scenes
# - Message about how the system works
# - Keep conversation history

# TO DO 28th of August 2026
# - Function to handle REAL END
# - change name to process_history
# - remove history for all LLMs except the conversation LLM
# - update the state keeping only the input parameters, but setting the process history back to null

# POSSIBLE PROBLEMS:
# - number of new max tokens too small and limits the generated output of the LLM

# QUESTIONS:
# - Python or LLM to convert values?

# NOTES:
# - LLM working better when it can "reason" and be wordy


model_id = "Models/Llama-3.1-8B-Instruct"

class InputParameters(BaseModel):

    mechanism: str | None = Field(default = None, description = "Chemical mechanism") #give a list of possible mechanisms? Database for the mechanisms?
    fuel: str | None = Field(default = None, description = "Fuel used for the combustion simulation.")
    pressure: float | None = Field(default = None, description = "Pressure at which the simulation is performed.")
    pressure_unit: str | None = Field(default = None, description = "Unit of the pressure provided by the user. E.g., 'P' for Pascal. 'atm' for standard atmosphere, 'bar'.")
    temperature: float | None = Field(default = None, description = "Temperature at which the simulation should be performed, keeping the number that is provided by the user. Do not convert to another unit.")
    temperature_unit: str | None = Field(default = None, description = "Unit of the temperature provided by the user. E.g., 'K' for Kelvin. 'C' for Celsius. 'F' for Fahrenheit.")
    equivalence_ratio: str | None = Field(default = None, description = "Equivalence ratio range")
    target_species: str | None = Field(default = None, description = "Target species")

schema = InputParameters.model_json_schema()

PROMPT_LLM_CHAT = """
                    You are the conversational assistant of a combustion mechanism selection system.

                    Your role is to communicate naturally with the user while helping them define the
                    input parameters required for a combustion mechanism selection/reduction task.

                    You are given:
                    1. The user's latest message.
                    2. A summary of what the system has done so far.
                    3. The current input parameters.

                    Your job is ONLY to produce the response that should be shown to the user.

                    GENERAL RULES:

                    - Be helpful, clear, concise, and natural.
                    - Answer the user's latest message directly.
                    - Do not invent facts, parameters, values, mechanisms, or experimental conditions.
                    - Do not change, extract, verify, or infer input parameters yourself.
                    - Treat the CURRENT INPUT PARAMETERS as the parameters currently established by the system.
                    - Treat the PROCESS HISTORY as a description of actions already performed by the system.
                    - Do not mention internal agents, LLMs, LangGraph, nodes, routing, prompts, JSON,
                    or other implementation details unless the user explicitly asks about how the
                    system works.
                    - Do not explain what another agent has done internally. Instead, communicate
                    the result naturally to the user.

                    WHEN PARAMETERS HAVE JUST BEEN RETRIEVED OR UPDATED:

                    - Clearly present the currently established parameters to the user when appropriate.
                    - If the system has filled or suggested values that were not explicitly provided
                    by the user, make it clear that these are suggested/default values rather than
                    values provided by the user.
                    - Ask the user whether the proposed configuration is correct when confirmation
                    is required.
                    - Do not silently present inferred or default values as if the user had provided them.

                    WHEN INFORMATION IS MISSING:

                    - Ask the user for the missing information that is necessary to continue.
                    - Prefer asking only the most relevant question(s), rather than listing many
                    questions at once.
                    - If several missing parameters are equally important, ask them in a logical order.
                    - Never invent an answer merely to avoid asking the user.

                    WHEN THE USER CORRECTS A PARAMETER:

                    - Acknowledge the correction naturally.
                    - Present the updated configuration if appropriate.
                    - If further confirmation is required, ask the user to confirm it.

                    WHEN THE USER ASKS A GENERAL COMBUSTION QUESTION:

                    - Answer the question normally if you can do so reliably.
                    - Do not modify the input parameters unless the user's message explicitly
                    requests a parameter change.
                    - If you are uncertain about a technical fact, say that you are uncertain rather
                    than inventing an answer.

                    CONFIRMATION:

                    When all required parameters are available and the system indicates that the
                    configuration should be confirmed, explicitly ask the user whether the complete
                    configuration is correct.

                    Do not assume that the user is satisfied merely because all parameters are filled.

                    STYLE:

                    - Professional but conversational.
                    - Concise.
                    - Avoid unnecessary technical jargon.
                    - Do not repeat information unnecessarily.
                    - Ask direct questions.
                    - Do not expose internal reasoning.

                    The response you generate will be shown directly to the user.
                    Therefore, output ONLY the natural-language response to the user.
                    """

PROMPT_LLM_RETRIEVE =   f"""
                        You are an information extraction agent.

                        Your task is to extract information from the user's message.
                        The user will describe a combustion problem and your task is to extract the relevant parameters.

                        Rules:
                        - Only extract information explicitly provided by the user.
                        - Never invent information.
                        - If information is not provided, use null.
                        - Return ONLY a JSON object.
                        - Do not add explanations or any text outside the JSON object.

                        The JSON object must follow this schema:

                        {json.dumps(schema, indent=2)}

                        IMPORTANT:
                        - Do NOT put the fields inside a "properties" object.
                        - "properties" in the description above only describes the available fields.
                        - Your final response must have the fields directly at the top level.

                        For example, the correct format is:

                        {{
                            "mechanism": null,
                            "fuel": null,
                            "pressure": null,
                            "pressure_unit": null,
                            "temperature": null,
                            "temperature_unit": null,
                            "equivalence_ratio": null,
                            "target_species": null
                        }}
                        """

PROMPT_LLM_VERIFY = f"""
                    You are a critical verification agent for a combustion simulation assistant.
                    Treat the message of the user as the ground truth and be critical with what the json contains.

                    Your task is to compare:

                    1. The original message written by the user.
                    2. The parameters extracted by another LLM.

                    Determine whether the extracted parameters are consistent with the information explicitly provided by the user.

                    Verification rules:
                    - Check every parameter individually.
                    - Only consider information explicitly stated by the user.
                    - Do not add information that the user did not provide.
                    - Do not assume missing values.
                    - Check that numerical values are copied correctly.
                    - Check that units are copied correctly.
                    - Check that the value and its unit are consistent.
                    - Pay particular attention to temperature and pressure units.
                    - If the extracted parameter is null and the user did not provide that parameter, this is correct.
                    - If the extracted parameter contains information that the user did not provide, this is incorrect.
                    - If a parameter differs from what the user explicitly stated, this is incorrect.
                    - If no information is provided and the current value is null, consider it as correct and keep it null.

                    If everything is correct, state that no modification is required.

                    If something is incorrect, clearly identify:
                    - which parameter is incorrect,
                    - what the extracted value is,
                    - what the user actually stated,
                    - what the corrected value should be.

                    Do not recommend values that the user did not provide.

                    Return a concise verification report.
                    """

PROMPT_LLM_UPDATE = f"""
                    You are a JSON parameter update agent for a combustion simulation assistant.

                    Your task is to update an existing JSON object based ONLY on an update instruction.

                    You are given:
                    1. CURRENT PARAMETERS: the parameters currently stored.
                    2. UPDATE INSTRUCTION: information describing what should be changed.

                    Your job is to modify ONLY the parameters that the update instruction explicitly requires.

                    IMPORTANT RULES:

                    1. The CURRENT PARAMETERS are the source of truth for all parameters that are
                    not being modified.

                    2. Preserve every existing parameter exactly as it is unless the update
                    instruction explicitly requires changing it.

                    3. Do NOT reset existing parameters to null.

                    4. Do NOT infer, estimate, calculate, or invent values.

                    5. Do NOT modify a parameter merely because it is mentioned in an explanation.
                    Modify it only when the update instruction explicitly indicates that it
                    should be changed.

                    6. If the update instruction changes one parameter, change only that parameter.

                    7. If the update instruction changes multiple parameters, change only those
                    parameters.

                    8. If the update instruction does not contain enough information to determine
                    a new value, keep the existing value unchanged.

                    9. For numerical values and units:
                    - Preserve the value exactly as specified by the update instruction.
                    - Do not convert units.
                    - Keep the value and its unit in their corresponding fields.
                    - For example, "200 degrees Celsius" means:
                        temperature = 200
                        temperature_unit = "C"
                    - "3k Celsius" means:
                        temperature = 3
                        temperature_unit = "C"
                        Do NOT convert this to 3000 K.

                    10. If a parameter is explicitly removed by the user, set that parameter and
                        its corresponding unit field to null.

                    11. Never modify parameters that are unrelated to the requested update.

                    12. The output must contain ALL fields from the schema, including fields that
                        were not modified.

                    13. Return ONLY a valid JSON object.
                        Do not return Markdown.
                        Do not return ```json.
                        Do not provide explanations.
                        Do not provide comments.

                    The JSON object must follow this schema:

                    {json.dumps(schema, indent=2)}

                    Example 1:

                    CURRENT PARAMETERS:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    UPDATE INSTRUCTION:
                    "Change the temperature to 1200 K."

                    CORRECT OUTPUT:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1200,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    Example 2:

                    CURRENT PARAMETERS:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    UPDATE INSTRUCTION:
                    "Actually, use ammonia instead of hydrogen."

                    CORRECT OUTPUT:
                    {{
                        "mechanism": null,
                        "fuel": "ammonia",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    Example 3:

                    CURRENT PARAMETERS:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    UPDATE INSTRUCTION:
                    "The user said 3k Celsius, but the extracted temperature was incorrectly
                    interpreted as 3000 K. Change the temperature to the value and unit actually
                    provided by the user."

                    CORRECT OUTPUT:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 3,
                        "temperature_unit": "C",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    Example 4:

                    CURRENT PARAMETERS:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 10,
                        "pressure_unit": "bar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    UPDATE INSTRUCTION:
                    "Change the pressure to 2000 mbar."

                    CORRECT OUTPUT:
                    {{
                        "mechanism": null,
                        "fuel": "hydrogen",
                        "pressure": 2000,
                        "pressure_unit": "mbar",
                        "temperature": 1000,
                        "temperature_unit": "K",
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    Now update the CURRENT PARAMETERS according to the UPDATE INSTRUCTION.

                    Return ONLY the complete updated JSON object.
                    """
PROMPT_LLM_FILL = f"""
                    You are the parameter completion agent of a combustion simulation assistant.

                    Your task is to complete a partially filled set of combustion simulation parameters.

                    You are given:
                    1. CURRENT PARAMETERS: a JSON object containing the parameters currently known.
                    2. Some parameters may have the value null because they are not yet known.

                    Your task is to provide a complete and plausible configuration by filling the missing parameters.

                    IMPORTANT:
                    This is a temporary default-completion agent. In the future, missing information
                    will be obtained from a database/RAG system. For now, use your general knowledge
                    of combustion simulations to provide reasonable default values.

                    RULES:

                    1. NEVER modify a parameter that already has a non-null value.
                    Preserve its value exactly.

                    2. Only fill parameters whose value is null.

                    3. For a missing parameter, choose a reasonable and commonly used value
                    for a combustion simulation.

                    4. Do not invent unusual, highly specific, or arbitrary values when a
                    conventional default is available.

                    5. For parameters involving a value and a unit:
                    - Fill the numerical value and its corresponding unit consistently.
                    - Do not convert or modify values that are already present.
                    - If both the value and unit are null, provide a reasonable value and unit.

                    6. If a reasonable value cannot be determined from the available information,
                    keep the parameter as null rather than making an arbitrary guess.

                    7. Do not add fields that are not part of the schema.

                    8. The output must contain ALL fields from the schema.

                    9. Return ONLY a valid JSON object.
                        Do not return Markdown.
                        Do not return ```json.
                        Do not provide explanations or comments.

                    The JSON object must follow this schema:

                    {json.dumps(schema, indent=2)}

                    IMPORTANT:
                    - Do NOT put the fields inside a "properties" object.
                    - "properties" in the description above only describes the available fields.
                    - Your final response must have the fields directly at the top level.

                    For example, the correct format is:

                    {{
                        "mechanism": null,
                        "fuel": null,
                        "pressure": null,
                        "pressure_unit": null,
                        "temperature": null,
                        "temperature_unit": null,
                        "equivalence_ratio": null,
                        "target_species": null
                    }}

                    Return the complete JSON object with the missing parameters filled where
                    a reasonable default can be provided.
                    """
PROMPT_LLM_ROUTER = """You are the routing agent of a combustion simulation assistant.

                    Your task is to determine what the assistant should do next based on:

                    1. The latest message from the user.
                    2. The parameters currently stored.

                    Available actions:

                    - RETRIEVE:
                    In case the input parameters are still empty, extract combustion parameters from the user's message and fill in the missign fields.

                    - UPDATE:
                    Modify parameters that have already been extracted in case the user indicates that he wants to modify one or multiple parameters.

                    - CHAT:
                    Respond conversationally to the user's message without modifying the simulation parameters in case the user wrote a message which is not related to the retrieval or update of certain input parameters.

                    - END:
                    In case all the input parameters are filled in and the user indicates that he is satisfied with the parameters, the conversation should end.

                    IMPORTANT:

                    - UPDATE is only available when existing input parameters are present.
                    - If no input parameters exist, never select UPDATE.
                    - If the user provides combustion parameters and no parameters currently
                    exist, select RETRIEVE.
                    - If parameters already exist and the user wants to change one or more
                    of them, select UPDATE.
                    - If the user asks a general question unrelated to modifying parameters,
                    select CHAT.
                    - If the user indicates that he is satisfied with all the parameters,
                    select END.

                    Return only the routing decision."""

# Fourth model for suggestions?


opening_message = "\nHello, I'm your combustion mechanism consultant.\nI will try to provide you the best chemical mechanism for your application.\nCan you describe the simulation you would like to perform?"

class ModelManager():

    def __init__(self, model_name: str):
        self.model_name = model_name
        
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self) -> None:
    
        self.tokenizer = AutoTokenizer.from_pretrained(
                                        self.model_name,
                                        clean_up_tokenization_spaces=False
                                    )

    def load_model(self) -> None:

        quantization_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16,
                                    )
        
        self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                quantization_config=quantization_config,
                                device_map="auto",
                            )

class AgentState(TypedDict):

    # Latest message from the user
    user_message: str

    # History of all the process
    history: list[str] | None

    # Current extracted parameters
    # None means that no parameters have been established yet
    input_parameters: InputParameters | None

    # Decision made by the router
    route: str | None

    # Final response to show to the user
    response: str | None

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

        selected_route = self.agent.LLM_router.define_route(state["user_message"], state["input_parameters"], possible_actions)

        return {"route": selected_route}

    def chat_node(self, state: AgentState):

        message = f"""
                    CURRENT USER MESSAGE:
                    {state["user_message"]}

                    PROCESS HISTORY:
                    {state["history"]}

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
                "history": state["history"] + [history_entry]}

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
                "history": state["history"] + [history_entry]}

    def fill_node(self, state: AgentState):

        LLM_fill_reply, input_parameters_filled = self.agent.LLM_fill.fill_missing_information(state["input_parameters"])
        
        print(f"\nAgent: {LLM_fill_reply}")

        history_entry = (
                        "FILL RESULT: The input parameters, that were missing from the user's message, have been filled based on the context provided by the user and combined with RAG retrieval on a combustion database. The agent should present the extracted parameters to the user and ask for confirmation."
                    )

        return {"input_parameters": input_parameters_filled,
                "history": state["history"] + [history_entry]}
    
    def route_after_router(self, state: AgentState):
        return state["route"]

    def route_after_retrieve(self, state: AgentState):
        if all(value is not None for value in state["input_parameters"].model_dump().values()):
            return "chat" #maybe don't use the chat in that case but directly print it? How to format the string in the history for the LLM
        
        return "fill"
    
class LLM:

    def __init__(self, model_manager: ModelManager, model_preprompt: str) -> None:

        self.model_manager = model_manager
        self.preprompt = model_preprompt
        # set temperature

        self.history = [{"role": "system", "content": model_preprompt}]

    def generate(self, message: str, max_new_tokens: int = 200, do_sample: bool = True) -> str:

        self.history.append({"role": "user", "content": message})

        inputs = self.model_manager.tokenizer.apply_chat_template(
                                self.history,
                                add_generation_prompt=True,
                                return_tensors="pt",
                            ).to(self.model_manager.model.device)

        outputs = self.model_manager.model.generate(  #set the temperature
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample = do_sample
                            )

        LLM_reply = self.model_manager.tokenizer.decode(
                                outputs[0][inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True,
                            )
        
        self.history.append({"role": "assistant", "content": LLM_reply})

        return LLM_reply

class LLM_conversation(LLM):

    def __init__(self, model_manager: ModelManager, model_preprompt: str, model_opening_message: str) -> None:

        super().__init__(model_manager, model_preprompt)

        self.history.append({"role": "assistant", "content": model_opening_message})

        #ask at the end if the configuration is correct?

class LLM_retrieval(LLM):

    def retrieve_information(self, message: str, max_new_tokens: int = 300) -> tuple[str, InputParameters]:

        #estimation for the uncertain parts?
        #set temperature

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample = False)

        #convert JSON -> Python dictionary
        data = json.loads(LLM_reply)

        #validate dictionary with Pydantic
        input_parameters = InputParameters.model_validate(data)

        return LLM_reply, input_parameters

class LLM_verify(LLM):

    def verify_information(self, user_message: str, input_parameters: InputParameters, max_new_tokens: int = 500) -> str:

        input_parameters_json = input_parameters.model_dump_json(indent=2)

        message = f"""ORIGINAL USER MESSAGE:
                        {user_message}

                        EXTRACTED PARAMETERS:
                        {input_parameters_json}

                        Verify whether the extracted parameters accurately represent the information
                        explicitly provided in the original user message."""

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample=False)

        return LLM_reply

class LLM_update(LLM):

    def update_information(self, message_update: str, current_input_parameters: str, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""UPDATE INSTRUCTION:
                    {message_update}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Update the current parameters according to the update instruction.
                    Return the complete updated JSON object.
                    """

        updated_json = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False)

        data = json.loads(updated_json)

        updated_input_parameters = InputParameters.model_validate(data)

        return updated_json, updated_input_parameters

class LLM_fill(LLM):

    def fill_missing_information(self, current_input_parameters: str, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Complete the missing parameters according to your instructions.
                    """

        filled_json = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False)

        print(filled_json)

        data = json.loads(filled_json)

        filled_input_parameters = InputParameters.model_validate(data)

        return filled_json, filled_input_parameters

class LLM_router(LLM):

    def define_route(self, user_message: str, current_input_parameters: str, list_of_possible_routes: list[str], max_new_tokens: int = 500):

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""USER MESSAGE:
                    {user_message}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Choose from the list below and output the most appropriate route based on the user message and the current set of parameters.

                    POSSIBLE ROUTES:
                    {list_of_possible_routes}

                    Only choose from this list. You cannot choose another option.
                    """

        chosen_route = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False)

        if chosen_route not in list_of_possible_routes:
            raise ValueError(f"LLM selected invalid route '{chosen_route}'. Expected one of: {list_of_possible_routes}.")

        return chosen_route
    
class Agent:

    def __init__(self, model_name: str, model_preprompts: list[str], model_opening_message: str) -> None:

        self.model_manager = ModelManager(model_name)
        
        self.LLM_conversation = LLM_conversation(self.model_manager, model_preprompts[0], model_opening_message)
        self.LLM_retrieval = LLM_retrieval(self.model_manager, model_preprompts[1])
        self.LLM_verification = LLM_verify(self.model_manager, model_preprompts[2])
        self.LLM_update = LLM_update(self.model_manager, model_preprompts[3])
        self.LLM_fill = LLM_fill(self.model_manager, model_preprompts[4])
        self.LLM_router = LLM_router(self.model_manager, model_preprompts[5])

        self.graph = AgentGraph(self)

    def chat(self) -> None:

        state = {
                "user_message": "",
                "history": [],
                "input_parameters": InputParameters(),
                "route": None,
                "response": None,
            }

        print(f"\nAgent: {opening_message}")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]: sys.exit()

                # Update only the part of the state that changes
                state["user_message"] = user_input

                # Run the LangGraph
                state = self.graph.app.invoke(state)

                # Display the response generated by the chat node
                print(f"\nAgent: {state['response']}")

            except KeyboardInterrupt:
                sys.exit()



chatting_agent = Agent(model_id, [PROMPT_LLM_CHAT, PROMPT_LLM_RETRIEVE, PROMPT_LLM_VERIFY, PROMPT_LLM_UPDATE, PROMPT_LLM_FILL, PROMPT_LLM_ROUTER], opening_message)
chatting_agent.chat()