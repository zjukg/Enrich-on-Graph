
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers

class opensource_llm_client:
    def __init__(self, model_path=None):
        config = AutoConfig.from_pretrained(model_path)
        print(f"Context window length: {config.max_position_embeddings}")

        # Using Qwen
        # Get tokenizer and model from pretrained model
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # # Use device_map for LLaMA to automatically allocate multi-GPU
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,  # Can switch to torch.float16 if GPU supports
        #     device_map="auto"  # Automatically allocate to multi-GPU or single GPU
        # )

        # Using llama
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )


    def response(self, question, mode):
        # Using Qwen
        # # Use model to generate reply
        # input_ids = self.tokenizer.encode(question, return_tensors="pt")
        
        # # Move input data to the device where model is located
        # input_ids = input_ids.to(self.model.device)

        # # Limit length when generating reply to avoid max_new_tokens being too high affecting performance
        # generated_ids = self.model.generate(
        #     input_ids,
        #     max_new_tokens=16000,  
        #     temperature=0.2,     # Adjust generation diversity
        #     top_p=1,           # Adjust generation quality
        # )

        # # Decode generated reply, filter special characters
        # response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # return response
        

        # Using llama
        messages = [
            # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": question},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=16000,
            temperature=0.2,     # Adjust generation diversity
            top_p=1,           # Adjust generation quality
        )
        return outputs[0]["generated_text"][-1]["content"]