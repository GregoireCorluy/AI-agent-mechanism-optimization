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

# POSSIBLE PROBLEMS:
# - number of new max tokens too small and limits the generated output of the LLM

# QUESTIONS:
# - Python or LLM to convert values?


model_id = "Models/Llama-3.1-8B-Instruct"

class InputParameters(BaseModel):

    mechanism: str | None = Field(default = None, description = "Chemical mechanism") #give a list of possible mechanisms? Database for the mechanisms?
    fuel: str | None = Field(default = None, description = "Fuel written in chemical notation. E.g. hydrogen = H2, ammonia = NH3, methane = CH4, mixture of hydrogen and ammonia = H2/NH3.")
    pressure: int | None = Field(default = None, description = "Pressure at which the simulation is performed expressed in standard atmosphere unit (atm). If another unit is given, do the conversion.") # (1 atm = 101325 Pa)
    temperature: float | None = Field(default = None, description = "Temperature at which the simulation should be performed expressed in Kelvin. If another unit is given, do the conversion.") #  (T (in degrees Celsius) -273 = ).
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
                            "fuel": "H2/NH3",
                            "pressure": null,
                            "temperature": 300,
                            "equivalence_ratio": null,
                            "target_species": null
                        }}
                        """

# Fourth model for suggestions?


opening_message = "Hello, I'm your assistant. How can I help you?"

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

class Agent:

    def __init__(self, model_name: str, model_preprompts: list[str], model_opening_message: str) -> None:

        self.model_manager = ModelManager(model_name)
        
        self.LLM_conversation = LLM_conversation(self.model_manager, model_preprompts[0], model_opening_message)
        self.LLM_retrieval = LLM_retrieval(self.model_manager, model_preprompts[1])

    def chat(self) -> None:

        print(f"\nAgent: {opening_message}")

        iteration = 0

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]: sys.exit()

                if iteration == 0:
                    LLM_reply, input_parameters = self.LLM_retrieval.retrieve_information(user_input)
                    #LLM_reply = self.LLM_retrieval.generate(user_input, max_new_tokens=300, do_sample=False)

                    print(f"\nAgent: {LLM_reply}")
                    print(f"Input parameters: {input_parameters}")

                else:
                    LLM_reply = self.LLM_conversation.generate(user_input)

                    print(f"\nAgent: {LLM_reply}")

                iteration += 1

            except KeyboardInterrupt:
                sys.exit()


chatting_agent = Agent(model_id, [PROMPT_LLM_CHAT, PROMPT_LLM_RETRIEVE], opening_message)
chatting_agent.chat()