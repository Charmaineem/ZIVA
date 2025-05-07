from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import LlamaCpp

class HuggingfaceModel:

    def __init__(self, model_path, model_type='huggingface'):
        if not model_path:
            raise ValueError('Missing model path')
        
        self.model_path = model_path
        self.model_type = model_type
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        if self.model_type == "huggingface":
            model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto")
            return model
        
        elif self.model_type == "llama-cpp":
            model = LlamaCpp(
                    model_path = self.model_path,
                    max_tokens=500,
                    seed=42,
                    verbose=False
                    )
        
    def load_tokenizer(self):
        if self.model_type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            return tokenizer
        
    def _generate(self,prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids']  # Access as dictionary key, not attribute
        
        # Rest of your code is correct
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=20
        )
        print(self.tokenizer.decode(output[0]))

# def main():
#     model = HuggingfaceModel(model_path="microsoft/Phi-3-mini-4k-instruct")
#     model._generate('What is 1+1?')

# if __name__ == "__main__":
#     main()