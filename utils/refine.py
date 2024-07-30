from transformers import pipeline

class TextRefiner:
    def __init__(self, access_token, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.text_generator = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            token = access_token,
            model_kwargs={"torch_dtype": "bfloat16"},
            device_map ='auto'
        )

    def refine(self, user_input):
        prompt_template = f"""
        I want you to transform the following user description into a structured text that contains a bunch of adjectives and rhetorics.
        If the user description is written in other languages than English, please translate it to English before refining.
        You should only give a structured text itself only (No need to give additional information).
        
        User description: {user_input}
        
        Structured description of user input: 
        """

        output = self.text_generator(prompt_template, max_length=200, do_sample=True, temperature=0.7)


        refined_text = output[0]['generated_text'].replace(prompt_template, "").strip()

        return refined_text

if __name__ == "__main__":
    refiner = TextRefiner(access_token='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')
    user_input = "A person is standing in front of a building."
    structured_text = refiner.refine(user_input)
    print(structured_text)