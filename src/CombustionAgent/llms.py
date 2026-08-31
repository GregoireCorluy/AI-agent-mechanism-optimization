from .models import ModelManager
from .parameters import InputParameters
import json

class LLM:

    def __init__(self, model_manager: ModelManager, model_preprompt: str) -> None:

        self.model_manager = model_manager
        self.preprompt = model_preprompt
        # set temperature

    def generate(self, message: str, max_new_tokens: int = 200, do_sample: bool = True) -> str:

        messages = [
            {"role": "system", "content": self.preprompt},
            {"role": "user", "content": message},
        ]

        inputs = self.model_manager.tokenizer.apply_chat_template(
                                                messages,
                                                add_generation_prompt=True,
                                                return_tensors="pt",
                                            ).to(self.model_manager.model.device)

        outputs = self.model_manager.model.generate( #add temperature?
                                                **inputs,
                                                max_new_tokens=max_new_tokens,
                                                do_sample=do_sample,
                                            )

        LLM_reply = self.model_manager.tokenizer.decode(
                                            outputs[0][inputs["input_ids"].shape[-1]:],
                                            skip_special_tokens=True,
                                        )

        return LLM_reply

class ConversationLLM(LLM):

    def __init__(self, model_manager: ModelManager, model_preprompt: str, model_opening_message: str) -> None:

        super().__init__(model_manager, model_preprompt)

        self.history = [{"role": "system", "content": model_preprompt},
                        {"role": "assistant", "content": model_opening_message},]

    def generate(self, message: str, max_new_tokens: int = 200, do_sample: bool = True) -> str:

        self.history.append({"role": "user", "content": message})

        inputs = self.model_manager.tokenizer.apply_chat_template(
                                self.history,
                                add_generation_prompt=True,
                                return_tensors="pt",
                            ).to(self.model_manager.model.device)

        outputs = self.model_manager.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=do_sample,
                            )

        LLM_reply = self.model_manager.tokenizer.decode(
                                outputs[0][inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True,
                            )

        self.history.append({"role": "assistant", "content": LLM_reply})

        return LLM_reply

class RetrievalLLM(LLM):

    def retrieve_information(self, message: str, max_new_tokens: int = 300) -> tuple[str, InputParameters]:

        #estimation for the uncertain parts?
        #set temperature

        LLM_reply = self.generate(message, max_new_tokens = max_new_tokens, do_sample = False)

        #convert JSON -> Python dictionary
        data = json.loads(LLM_reply)

        #validate dictionary with Pydantic
        input_parameters = InputParameters.model_validate(data)

        return LLM_reply, input_parameters

class VerifyLLM(LLM):

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

class UpdateLLM(LLM):

    def update_information(self, message_update: str, current_input_parameters: InputParameters, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

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

class FillLLM(LLM):

    def fill_missing_information(self, current_input_parameters: InputParameters, max_new_tokens: int = 500) -> tuple[str, InputParameters]:

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

class RouterLLM(LLM):

    def define_route(self, agent_message: str, user_message: str, current_input_parameters: InputParameters, list_of_possible_routes: list[str], max_new_tokens: int = 500):

        current_input_parameters_json = current_input_parameters.model_dump_json(indent=2)

        message = f"""LAST AGENT MESSAGE:
                    {agent_message}
        
                    USER MESSAGE:
                    {user_message}

                    CURRENT PARAMETERS:
                    {current_input_parameters_json}

                    Choose from the list below and output the most appropriate route based on the last agent message, the user message and the current set of parameters.

                    POSSIBLE ROUTES:
                    {list_of_possible_routes}

                    Only choose from this list. You cannot choose another option.
                    """

        chosen_route = self.generate(message, max_new_tokens=max_new_tokens, do_sample=False)

        if chosen_route not in list_of_possible_routes:
            raise ValueError(f"LLM selected invalid route '{chosen_route}'. Expected one of: {list_of_possible_routes}.")

        return chosen_route