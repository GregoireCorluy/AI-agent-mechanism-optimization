import torch

from transformers import (
                    AutoTokenizer,
                    AutoModelForCausalLM,
                    BitsAndBytesConfig,
                )

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