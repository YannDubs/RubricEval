from pathlib import Path
from typing import Sequence

from alpaca_eval import utils as ae_utils
import ast
import pandas as pd

from alpaca_eval.annotators import base

__all__ = ["Instructionator2", "Instructionator", "Rubricator", "Completor", "Evaluator"]


class Instructionator2(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the initial instruction.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "instructionator_configs_2"

    def __init__(
            self,
            *args,
            primary_keys: Sequence[str] = ("n", "category"),
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
        return "categories_and_instructions"


class Instructionator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the initial instruction.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "instructionator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("batch_idx",),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(*args, annotators_config=annotators_config, primary_keys=primary_keys, **kwargs)

    @property
    def annotation_key(self) -> str:
        return "categories_and_instructions"

    def generate_n_instructions_per_categories(self, n: int) -> Sequence[dict]:
        """Generate n instructions per categories."""
        df_input = pd.Series(range(n), name="batch_idx").astype(str).to_frame()
        instructions = self(df_input)
        df_instructions = ae_utils.convert_to_dataframe(instructions)
        df_instructions[self.annotation_key] = df_instructions[self.annotation_key].apply(ast.literal_eval)
        df_instructions = df_instructions.explode(column=[self.annotation_key]).reset_index(drop=True)
        df_instructions = pd.concat(
            [
                df_instructions.drop([self.annotation_key], axis=1),
                pd.json_normalize(df_instructions[self.annotation_key]),
            ],
            axis=1,
        )
        return df_instructions.to_dict(orient="records")


class Rubricator(base.BaseAnnotatorJSON):
    __doc__ = base.BaseAnnotatorJSON.__doc__.replace(
        "Base class for a pool of annotators.",
        "Auto annotator for writing the rubric.",
    )
    TMP_MISSING_ANNOTATION = "TMP_MISSING_ANNOTATION"
    DEFAULT_ANNOTATION_TYPE = str
    DEFAULT_BASE_DIR = Path(__file__).parent / "rubricator_configs"

    def __init__(
        self,
        *args,
        primary_keys: Sequence[str] = ("prompt",),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(*args, annotators_config=annotators_config, primary_keys=primary_keys, **kwargs)

    @property
    def annotation_key(self) -> str:
        return "assignment_and_rubric"

    def make_df_rubrics(self, annotated : Sequence[dict]) -> pd.DataFrame:
        df_rubrics = ae_utils.convert_to_dataframe(annotated)
        df_rubrics = pd.concat(
            [
                df_rubrics.drop([self.annotation_key], axis=1),
                pd.json_normalize(df_rubrics[self.annotation_key].apply(ast.literal_eval), max_level=0),
            ],
            axis=1,
        )
        return df_rubrics


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
        super().__init__(*args, annotators_config=annotators_config, primary_keys=primary_keys, **kwargs)

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
        primary_keys: Sequence[str] = ("final_prompt", "detailed_analytic_rubric", "output", "key_criteria", "scoring_scales"),
        annotators_config="gpt4_CoT_v0",
        **kwargs,
    ):
        super().__init__(*args, annotators_config=annotators_config, primary_keys=primary_keys, **kwargs)

    @property
    def annotation_key(self) -> str:
        return "criteria_scores"
