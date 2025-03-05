import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer
)

class InferlessPythonModel:
    def initialize(self):
        model_name = "tenyx/Llama3-TenyxChat-70B"
        snapshot_download(repo_id=model_name,allow_patterns=["*.safetensors"])
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": 0})
        
    def infer(self, inputs):
        prompt = inputs["prompt"]
        chat = [{"role": "user", "content": prompt}]
        chat_template = self.tokenizer.apply_chat_template(chat,tokenize=False)
        inputs = self.tokenizer(chat_template,return_tensors="pt")
        generated_output = self.model.generate(**inputs, max_new_tokens=120)
        output = self.tokenizer.decode(generated_output[0], skip_special_tokens=True)
        return {"generated_outputs":output}
        
    def finalize(self,args):
        self.model = None
