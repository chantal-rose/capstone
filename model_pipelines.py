from functools import lru_cache
import re
from typing import Union

from transformers import (AutoModelForCausalLM,
                          AutoModelForQuestionAnswering,
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          BioGptForCausalLM,
                          BioGptTokenizer)
from transformers import pipeline, set_seed


MODEL = "model"
MODEL_NAME = "model_name"
QUESTION_ANSWERING = "question-answering"
TASK = "task"
TEXT_CLASSIFICATION = "text-classification"
TEXT_GENERATION = "text-generation"
TEXT2TEXT_GENERTAION = "text2text-generation"
TOKENIZER = "tokenizer"


# TODO: Model inference on gpu is faster, consider setting a device in pipeline


@lru_cache()
def load_models():  # pragma: no cover
    models = {
        "microsoft/biogpt": {
            TOKENIZER: BioGptTokenizer.from_pretrained("microsoft/biogpt"),
            MODEL: BioGptForCausalLM.from_pretrained("microsoft/biogpt"),
            TASK: TEXT_GENERATION
        },
        "akdeniz27/deberta-v2-xlarge-cuad": {
            TOKENIZER: AutoTokenizer.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            TASK: QUESTION_ANSWERING
        },
        # "AlexWortega/taskGPT2-xl-v0.2a": {
        #     TOKENIZER: AutoTokenizer.from_pretrained("AlexWortega/taskGPT2-xl-v0.2a"),
        #     MODEL: AutoModelForCausalLM.from_pretrained("AlexWortega/taskGPT2-xl-v0.2a"),
        #     TASK: TEXT_GENERATION
        # },
        "mrm8488/longformer-base-4096-finetuned-squadv2": {
            TOKENIZER: AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2"),
            TASK: QUESTION_ANSWERING
        },
        "allenai/longformer-large-4096-finetuned-triviaqa": {
            TOKENIZER: AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa"),
            TASK: QUESTION_ANSWERING
        },
        "Sarmila/pubmed-bert-squad-covidqa": {
            TOKENIZER: AutoTokenizer.from_pretrained("Sarmila/pubmed-bert-squad-covidqa"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("Sarmila/pubmed-bert-squad-covidqa"),
            TASK: QUESTION_ANSWERING
        },
        "vanadhi/roberta-base-fiqa-flm-sq-flit": {
            TOKENIZER: AutoTokenizer.from_pretrained("vanadhi/roberta-base-fiqa-flm-sq-flit"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("vanadhi/roberta-base-fiqa-flm-sq-flit"),
            TASK: QUESTION_ANSWERING
        },
        "Rakib/roberta-base-on-cuad": {
            TOKENIZER: AutoTokenizer.from_pretrained("Rakib/roberta-base-on-cuad"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("Rakib/roberta-base-on-cuad"),
            TASK: QUESTION_ANSWERING
        },
        "ixa-ehu/SciBERT-SQuAD-QuAC": {
            TOKENIZER: AutoTokenizer.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC"),
            TASK: QUESTION_ANSWERING
        },
        "razent/SciFive-base-Pubmed_PMC": {
            TOKENIZER: AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed_PMC"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed_PMC"),
            TASK: TEXT_CLASSIFICATION
        },
        "MaRiOrOsSi/t5-base-finetuned-question-answering": {
            TOKENIZER: AutoTokenizer.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            TASK: TEXT2TEXT_GENERTAION
        },
        "ozcangundes/T5-base-for-BioQA": {
            TOKENIZER: AutoTokenizer.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            TASK: QUESTION_ANSWERING
        }
    }
    return models


def load_pipeline(models: dict, model_dict: dict) -> pipeline:
    """Loads a pipeline object to be used by a model for inference

    :param models: Dictionary mapping models to their tokenizers and model classes
    :param model_dict: The model json which contains important information about the model
    :return Pipeline object
    """
    model_name = model_dict[MODEL_NAME]
    return pipeline(models[model_name][TASK], model=models[model_name][MODEL], tokenizer=models[model_name][TOKENIZER])


def get_answer_from_model(pipe: pipeline,
                          models: dict,
                          model_dict: dict,
                          question: str,
                          context: Union[str, None]) -> tuple:
    """Gets an answer from a model.

    The way the pipeline object is used depends on the task being performed.

    :param pipe: Pipeline object
    :param models: Dictionary mapping models to their tokenizers and model classes
    :param model_dict: The model json which contains important information about the model
    :param question: Question
    :param context: Context
    :return: Tuple of answer and score
    :raise Exception if task not supported
    """
    set_seed(42)
    model_name = model_dict[MODEL_NAME]
    task = models[model_name][TASK]
    if task == QUESTION_ANSWERING:
        output = pipe(question=question, context=context)
        return output["answer"], output["score"]
    elif task == TEXT_GENERATION or task == TEXT2TEXT_GENERTAION:
        text = "question: {} context: {}".format(question, context)
        pattern = re.compile(r".*answer: (.+)")
        output = pipe(text, max_length=50, num_return_sequences=1, do_sample=True)
        return pattern.match(output[0]["generated_text"]).groups()[0], None
    elif task == TEXT_CLASSIFICATION:
        output = pipe(question)
        return output[0]["label"], output[0]["score"]
    else:
        raise Exception("Task not supported by pipeline.")
