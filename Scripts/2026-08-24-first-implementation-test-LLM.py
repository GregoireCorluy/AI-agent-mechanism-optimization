import torch
import sys

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

model_id = "Models/Llama-3.1-8B-Instruct"

PROMPT_AGENT = "You're a helpful and kind assistant who tries to answer correctly to the questions of the user. " \
                "If you're not sure, don't give an answer and say that you don't know the answer."

opening_message = "Hello, I'm your assistant. How can I help you?"

class LLM:

    def __init__(self, model_name: str, model_preprompt: str, model_opening_message: str):

        self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name,
                                clean_up_tokenization_spaces=False
                            )

        quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                            )

        self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=quantization_config,
                                device_map="auto",
                            )
        
        self.preprompt = model_preprompt
        self.opening_message = model_opening_message

        self.history = [{"role": "system", "content": model_preprompt},
                        {"role": "assistant", "content": model_opening_message}]

    def generate(self, message, max_new_tokens: int = 200):

        self.history.append({"role": "user", "content": message})

        inputs = self.tokenizer.apply_chat_template(
                                self.history,
                                add_generation_prompt=True,
                                return_tensors="pt",
                            ).to(self.model.device)

        outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                            )

        return self.tokenizer.decode(
                                outputs[0][inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True,
                            )

    def chat(self):

        print(f"\nAgent: {opening_message}")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]: sys.exit()

                agent_reply = self.generate(user_input)

                print(f"\nAgent: {agent_reply}")
                self.history.append({"role": "assistant", "content": agent_reply})

            except KeyboardInterrupt:
                sys.exit()


LLM_test = LLM(model_id, PROMPT_AGENT, opening_message)
LLM_test.chat()