from transformers import pipeline
import os


class TextRefiner:
    def __init__(self):
        os.makedirs('./gpt2_cache', exist_ok=True)
        self.generator = pipeline('text-generation', model='gpt2', model_kwargs={'cache_dir': './gpt2_cache'})
        
        with open('prompt.txt', 'r') as f:
            self.prompt = f.read()
        
    def refine(self, user_input):
        full_prompt = self.prompt + user_input + " ->"
        generated_text = self.generator(full_prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.7)[0]['generated_text'].split(full_prompt)[1].split('\n')[0].strip()
        return generated_text