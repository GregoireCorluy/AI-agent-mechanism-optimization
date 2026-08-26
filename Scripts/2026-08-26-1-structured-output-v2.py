import torch
import sys
import json
from pydantic import BaseModel, Field
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

PROMPT_LLM_CHAT = "You're a helpful and kind assistant who tries to answer correctly to the questions of the user. " \
                "If you're not sure, don't give an answer and say that you don't know the answer."

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

    def verify_information(self, user_message: str, input_parameters_json: str, max_new_tokens: int = 500) -> str:

        message = f"""ORIGINAL USER MESSAGE:
                        {user_message}

                        EXTRACTED PARAMETERS:
                        {input_parameters_json}

                        Verify whether the extracted parameters accurately represent the information
                        explicitly provided in the original user message."""

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample=False)

        return LLM_reply

class LLM_update(LLM):

    def update_information(self, message_update: str, current_input_parameters_json: str, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

        message = f"""UPDATE INSTRUCTION:
                    {message_update}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Update the current parameters according to the update instruction.
                    Return the complete updated JSON object.
                    """

        updated_json = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False)

        data = json.loads(updated_json)

        input_parameters = InputParameters.model_validate(data)

        return updated_json, input_parameters

class Agent:

    def __init__(self, model_name: str, model_preprompts: list[str], model_opening_message: str) -> None:

        self.model_manager = ModelManager(model_name)
        
        self.LLM_conversation = LLM_conversation(self.model_manager, model_preprompts[0], model_opening_message)
        self.LLM_retrieval = LLM_retrieval(self.model_manager, model_preprompts[1])
        self.LLM_verification = LLM_verify(self.model_manager, model_preprompts[2])
        self.LLM_update = LLM_update(self.model_manager, model_preprompts[3])

    def chat(self) -> None:

        print(f"\nAgent: {opening_message}")

        iteration = 0

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]: sys.exit()

                if iteration == 0:
                    LLM_retrieval_reply, input_parameters = self.LLM_retrieval.retrieve_information(user_input)
                    #LLM_reply = self.LLM_retrieval.generate(user_input, max_new_tokens=300, do_sample=False)

                    print(f"\nAgent: {LLM_retrieval_reply}")
                    print(f"Input parameters: {input_parameters}")

                    LLM_verification_reply = self.LLM_verification.verify_information(user_input, LLM_retrieval_reply)

                    print(LLM_verification_reply)

                    LLM_reply, input_parameters_updated = self.LLM_update.update_information(LLM_verification_reply, LLM_retrieval_reply)
                    
                    print(f"\nAgent: {LLM_reply}")

                else:
                    #LLM_reply = self.LLM_conversation.generate(user_input)
                    LLM_reply, input_parameters_updated = self.LLM_update.update_information(user_input, input_parameters_updated)

                    print(f"\nAgent: {LLM_reply}")

                iteration += 1

            except KeyboardInterrupt:
                sys.exit()


chatting_agent = Agent(model_id, [PROMPT_LLM_CHAT, PROMPT_LLM_RETRIEVE, PROMPT_LLM_VERIFY, PROMPT_LLM_UPDATE], opening_message)
chatting_agent.chat()