from pathlib import Path
from typing import Sequence, Dict

from alpaca_eval import utils as ae_utils
import ast
import pandas as pd

from alpaca_eval.annotators import base

__all__ = ["Instructionator", "RubricBrainstormer", "RubricGenerator", "Rubricator", "Completor", "Evaluator"]


class Instructionator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the initial instruction.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "instructionator_configs"

    CATEGORIES: Dict[str, str] = {
        "Creativity": "This involves generating novel, useful, and unexpected ideas or concepts. This can be assessed through tasks such as story generation, poem writing, creating new game rules, developing innovative product ideas, etc.",
        "Math & Logical Reasoning": "This involves the capacity to understand, analyze, and solve problems using mathematical concepts, formal logic, and deductive reasoning. This can be assessed through tasks such as solving math problems, logical puzzles, proofs, etc.",
        "Coding": "Check the ability to understand, generate, design and debug code in various programming languages. This can include tasks such as writing algorithms, debugging code, refactoring software, manually executing code. writing unit tests, etc.",
        "Factual Knowledge": "This relates to the accurate recall and understanding of concrete facts and information, often about the world, history, geography, science, culture, art, etc. This should be agnostic to parameters that might change in the future (date, current position such as president, etc.).",
        "Common Sense Reasoning": "This involves applying basic, intuitive understanding of everyday situations and events, often involving implicit knowledge that humans typically take for granted. This can be assessed through tasks such as solving riddles, answering common sense questions, predicting the outcome of everyday events or making a plan of actions to perform a task.",
        "Task Completion": "Evaluate the model's ability to carry out specific tasks, such as summarizing a text, changing the format of a document, classifying data, etc.",
        "Adaptability to Different Domains & Role Playing": "Assess how well the model can understand and generate text related to specific domains, such as medicine, law, finance, science, etc.",
        "Ethical Reasoning": "This refers to the capacity to consider, evaluate, and make decisions based on moral principles and ethical guidelines. This can be assessed through tasks such as ethical dilemmas, moral reasoning, and ethical decision-making. This should not be biased, i.e. depend on cultural or religious beliefs.",
        "Emotional Intelligence": "This involves understanding, interpreting, and responding to emotions in oneself and others, often linked to empathy, self-awareness, and social skills. This can be assessed through tasks such as analyzing the emotion of the writer, generating empathetic responses, understanding the emotional state of a character, etc.",
        "Multi-language Proficiency": "Evaluate how well the model performs across different languages, not just in terms of fluency, but also in understanding cultural nuances, idioms, and colloquialisms. This can be assessed through tasks such as translation, understanding of cultural references, idiomatic expressions, etc.",
        "Robustness": "Test the model's ability to understand and respond appropriately even in the presence of noise(e.g. typos, grammatical errors, ...), conflicting pieces of information, ambiguity, missing or irrelevant information, and other adversarial inputs.",
        "Argumentation": "This involves the capacity to construct and evaluate arguments, often based on evidence and reasoning. This can be assessed through tasks such as writing persuasive essays, debating, evaluating the strength of an argument, etc.",
        "Analyzing & Interpreting Data": "This involves the capacity to understand, analyze, and interpret data, often using statistical methods and tools. This can be assessed by generating data, articles or reports and asking the model to analyze and interpret the data in various ways.",
    }

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("n", "category", "categories"),
        annotators_config: str = "gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "categories_and_instructions"

    def _generate_list(self) -> str:
        return "\n".join(
            [f" - {key}: {value}" for key, value in self.CATEGORIES.items()]
        )

    def generate_n_instruction_for_a_category(
        self, n: int, category: str
    ) -> Sequence[dict]:
        input = [{"n": n, "category": category, "categories": self._generate_list()}]
        return self(input)


class BaseRubricator(base.BaseAnnotatorJSON):
    DEFAULT_ANNOTATION_TYPE = str
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    
    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("prompt",),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    def make_df_rubrics(self, annotated: Sequence[dict]) -> pd.DataFrame:
        df_rubrics = ae_utils.convert_to_dataframe(annotated)
        df_rubrics = pd.concat(
            [
                df_rubrics.drop([self.annotation_key], axis=1),
                pd.json_normalize(
                    df_rubrics[self.annotation_key].apply(ast.literal_eval), max_level=0
                ),
            ],
            axis=1,
        )
        return df_rubrics
    

class RubricBrainstormer(BaseRubricator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for brainstorming criteria.",
    )
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubric_brainstormer_configs"

    @property
    def annotation_key(self) -> str:
        return "criteria"
    

class RubricGenerator(BaseRubricator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for generating detailed rubrics.",
    )
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubric_generator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("final_prompt", "clear_goals", "criteria"),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "detailed_rubric"


class Rubricator(BaseRubricator):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the rubric.",
    )
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubricator_configs"

    @property
    def annotation_key(self) -> str:
        return "assignment_and_rubric"


# TODO: should likely be in a different file and not inherit from BaseAnnotatorJSON.
class Completor(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Model to evaluate.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "completor_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("final_prompt",),
        annotators_config="claude-2",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "output"


class Evaluator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto evaluator of the output using the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = "object"  # use object to have dict output
    DEFAULT_BASE_DIR = Path(__file__).parent / "evaluator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = (
            "final_prompt",
            "detailed_analytic_rubric",
            "output",
            "criteria",
            "scoring_scales",
        ),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(
            *args,
            annotators_config=annotators_config,
            primary_keys=primary_keys,
            **kwargs,
        )

    @property
    def annotation_key(self) -> str:
        return "criteria_scores"
