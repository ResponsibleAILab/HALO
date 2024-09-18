def load_prompts(fewshot_prompt_file, cot_prompt_file):
    with open(fewshot_prompt_file, 'r') as file:
        fewshot_prompt = file.read()

    with open(cot_prompt_file, 'r') as file:
        cot_prompt = file.read()

    return fewshot_prompt, cot_prompt
