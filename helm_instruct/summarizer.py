import tiktoken

class Summarizer():
    """Summarizes instruction-specific feedback.

    Parameters:
    ----------
    all_instructions : DataFrame
        Instructions DataFrame to generate summaries from (e.g., output from RubricEval's Evaluator).

    category_col: str
        Column in DataFrame that stores the category of each instruction.

    config_name: str
        RubricEval Completor configuration folder name. Folder must contain a "configs.yaml" and "prompt.txt".

    model_name: str, optional
        Model to use for tokenization. Must be an OpenAI model and should match the Completor config. Default: "gpt-4o".

    split_size: int, optional
        Max token count allocated for category summary prompts. If all instructions can't fit within this limit, input is chunked and category summaries are "summaries of summaries". Default: 32000.
    """

    def __init__(
        self,
        all_instructions,
        category_col,
        config_name,
        model_name = "gpt-4o",
        split_size = 32000,
    ):
        self.all_instructions = all_instructions
        self.category_col = category_col
        self.config_name = config_name
        self.model_name = model_name
        self.split_size = split_size
        self.enc = tiktoken.encoding_for_model(self.model_name)

    def create_chunks(self, df):
        chunk_list = []
        chunk = pd.DataFrame()
        num_tokens = 0

        for _, row in df.iterrows():        
            if num_tokens + row["tokens"] > self.split_size:
                chunk_list.append(chunk)
                chunk = pd.DataFrame()
                num_tokens = 0

            chunk = pd.concat([chunk, pd.DataFrame([row])], ignore_index = True)
            num_tokens += row["tokens"]

        if not chunk.empty:
            chunk_list.append(chunk)

        return chunk_list

    def get_tokens(self, df):
        token_list = []

        for i, r in df.iterrows():
            text = f"""
    # Example {i}

    ## Category
    {r['category']}

    ## Prompt
    {r['final_prompt']}

    ## Rubric
    {yaml.dump(r['detailed_analytic_rubric'], default_flow_style=False)}

    ## Criteria scores
    {yaml.dump(r['criteria_scores'], default_flow_style=False)}

    ## Expert feedback
    {yaml.dump(r['feedback'], default_flow_style=False)}
        """
            
            tokens = len(self.enc.encode(text))
            token_list.append(tokens)

        return token_list

    def get_summary_completion(self, prompt):
        result = get_completions(pd.DataFrame({"final_prompt": [prompt]}), self.config_name)
        completion = result[0]["raw_completion"]
        return completion
        
    def get_instruction_summary_prompt(self, instructions_to_summarize):
        feedback = ""

        for i, r in instructions_to_summarize.iterrows():
            feedback += f"""
    # Example {i}

    ## Category
    {r['category']}

    ## Prompt
    {r['final_prompt']}

    ## Rubric
    {yaml.dump(r['detailed_analytic_rubric'], default_flow_style=False)}

    ## Criteria scores
    {yaml.dump(r['criteria_scores'], default_flow_style=False)}

    ## Expert feedback
    {yaml.dump(r['feedback'], default_flow_style=False)}
            """

        n_to_eval = len(instructions_to_summarize)
        prompt = f"""
    <|im_start|>system
    You are a helpful assistant that summarizes the strengths and weaknesses of answers of a model. 
    In particular we evaluated the answers of a model on {n_to_eval} instructions, and have scored each of them with a separate rubric. 
    We now want a simple aggregate summary of what the model does well and bad. 
    The summary should be very concise, use bullet points, and follow a very specific structure: first explain the strengths of the model, then explain the weaknesses of the model. Here's an example of that structure:
    Strenghts:
    - Strength 1
    - Strength 2
    ...
    Weaknesses:
    - Weakness 1
    - Weakness 2
    ...
    Make sure you fully adhere to this structure without any deviations.
    The strengths and weaknesses of the model you identify should be trends you notice across a number of instructions.
    Make sure that for each strength and weakness you identify, you provide some reasoning as to why it is a strength or weakness (for example, by explaining specific instances where the model did something right or wrong).
    The summary will be read by a user that wants to know whether they can use the model for their application.
    Remember that you MUST include specific examples of instructions when you discuss strengths or weaknesses. For example, when you discuss a strength, include what sort of instructions exactly that relates to in a way that anyone can understand.
    HOWEVER, remember that the user doesn't know the instructions (so don't refer to examples by number or the word "the"). Don't use bold typesetting.
    <|im_end|>
    <|im_start|>user
    {feedback}
    <|im_end|>
            """

        return prompt

    def get_single_category_summary_prompt(self, chunks, category):
        feedback = ""
        for i, r in enumerate(chunks):
            feedback += f"""
    # Summary {i}

    ## Category
    {category}

    ## Summary
    {r}
            """

        prompt = f"""
    <|im_start|>system
    You are a helpful assistant that summarizes the strengths and weaknesses of a model.
    We have already generated {len(chunks)} separate summaries, all of which summarize the model's abilities in the category of {category}.
    We now want a simple aggregate summary of what the model does well and bad in the category of {category}.
    The summary should be very concise and use bullet points, just like the individual summaries are. 
    The summary will be read by a user that wants to know whether they can use the model for their application.
    <|im_end|>
    <|im_start|>user
    {feedback}
    <|im_end|>
            """

        return prompt

    def get_multi_category_summary_prompt(self, categories, category_summaries):
        feedback = ""
        for i, r in enumerate(category_summaries):
            feedback += f"""
    # Summary {i}

    ## Category
    {categories[i]}

    ## Summary
    {r}
            """

        prompt = f"""
    <|im_start|>system
    You are a helpful assistant that summarizes the strengths and weaknesses of a model.
    We have already generated {len(category_summaries)} separate summaries, each of which summarizes the model's abilities in a different category.
    We now want a simple aggregate summary of the most important things the model does well and bad across all categories.
    The summary should be very concise and use bullet points, just like the individual category summaries are. 
    The summary will be read by a user that wants to know whether they can use the model for their application.
    <|im_end|>
    <|im_start|>user
    {feedback}
    <|im_end|>
            """

        return prompt

    def summarize_single_set(self, instructions):
        return self.get_summary_completion(self.get_instruction_summary_prompt(instructions))

    def summarize_category(self, category):
        category_instructions = self.all_instructions[self.all_instructions[self.category_col] == category]
        chunk_summaries = [self.summarize_single_set(chunk) for chunk in create_chunks(category_instructions)]
        return self.get_summary_completion(self.get_single_category_summary_prompt(chunk_summaries, category))

    def summarize_all(self):
        self.all_instructions["tokens"] = get_tokens(self.all_instructions)

        categories = []
        category_summaries = []

        for category in list(self.all_instructions.drop_duplicates(subset = [self.category_col])[self.category_col]):
            categories.append(category)
            category_summaries.append(self.summarize_category(category))

        overall_summary = self.get_summary_completion(self.get_multi_category_summary_prompt(categories, category_summaries))

        return (categories, category_summaries, overall_summary)
