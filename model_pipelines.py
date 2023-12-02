from functools import lru_cache
import re
from typing import Union

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForQuestionAnswering,
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          T5Tokenizer,
                          T5ForConditionalGeneration
                          )
from transformers import pipeline, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "akdeniz27/deberta-v2-xlarge-cuad": {
            TOKENIZER: AutoTokenizer.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            TASK: QUESTION_ANSWERING
        },
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
            TASK: TEXT_GENERATION
        },
        "MaRiOrOsSi/t5-base-finetuned-question-answering": {
            TOKENIZER: AutoTokenizer.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            TASK: TEXT2TEXT_GENERTAION
        },
        "ozcangundes/T5-base-for-BioQA": {
            TOKENIZER: T5Tokenizer.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            MODEL: T5ForConditionalGeneration.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            TASK: QUESTION_ANSWERING
        }
    }
    return models


@lru_cache()
def load_model(model_name):
    if model_name == 'akdeniz27/deberta-v2-xlarge-cuad':
        model = {"akdeniz27/deberta-v2-xlarge-cuad": {
            TOKENIZER: AutoTokenizer.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("akdeniz27/deberta-v2-xlarge-cuad"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "mrm8488/longformer-base-4096-finetuned-squadv2":
        model = {"mrm8488/longformer-base-4096-finetuned-squadv2": {
            TOKENIZER: AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "allenai/longformer-large-4096-finetuned-triviaqa":
        model = {"allenai/longformer-large-4096-finetuned-triviaqa": {
            TOKENIZER: AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "Sarmila/pubmed-bert-squad-covidqa":
        model = {"Sarmila/pubmed-bert-squad-covidqa": {
            TOKENIZER: AutoTokenizer.from_pretrained("Sarmila/pubmed-bert-squad-covidqa"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("Sarmila/pubmed-bert-squad-covidqa"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "vanadhi/roberta-base-fiqa-flm-sq-flit":
        model = {"vanadhi/roberta-base-fiqa-flm-sq-flit": {
            TOKENIZER: AutoTokenizer.from_pretrained("vanadhi/roberta-base-fiqa-flm-sq-flit"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("vanadhi/roberta-base-fiqa-flm-sq-flit"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "Rakib/roberta-base-on-cuad":
        model = {"Rakib/roberta-base-on-cuad": {
            TOKENIZER: AutoTokenizer.from_pretrained("Rakib/roberta-base-on-cuad"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("Rakib/roberta-base-on-cuad"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "ixa-ehu/SciBERT-SQuAD-QuAC":
        model = {"ixa-ehu/SciBERT-SQuAD-QuAC": {
            TOKENIZER: AutoTokenizer.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC"),
            MODEL: AutoModelForQuestionAnswering.from_pretrained("ixa-ehu/SciBERT-SQuAD-QuAC"),
            TASK: QUESTION_ANSWERING
        }}
    elif model_name == "razent/SciFive-base-Pubmed_PMC":
        model = {"razent/SciFive-base-Pubmed_PMC": {
            TOKENIZER: AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed_PMC"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed_PMC"),
            TASK: TEXT_GENERATION
        }}
    elif model_name == "MaRiOrOsSi/t5-base-finetuned-question-answering":

        model = {"MaRiOrOsSi/t5-base-finetuned-question-answering": {
            TOKENIZER: AutoTokenizer.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            MODEL: AutoModelForSeq2SeqLM.from_pretrained("MaRiOrOsSi/t5-base-finetuned-question-answering"),
            TASK: TEXT2TEXT_GENERTAION
        }}
    elif model_name == "ozcangundes/T5-base-for-BioQA":
        model = {"ozcangundes/T5-base-for-BioQA": {
            TOKENIZER: T5Tokenizer.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            MODEL: T5ForConditionalGeneration.from_pretrained("ozcangundes/T5-base-for-BioQA"),
            TASK: TEXT_GENERATION
        }}
    return model


def load_pipeline(models: dict, model_dict: dict) -> pipeline:
    """Loads a pipeline object to be used by a model for inference

    :param models: Dictionary mapping models to their tokenizers and model classes
    :param model_dict: The model json which contains important information about the model
    :return Pipeline object
    """
    #device = "cpu"
    
    model_name = model_dict[MODEL_NAME]
    return pipeline(models[model_name][TASK], model=models[model_name][MODEL], tokenizer=models[model_name][TOKENIZER], device=device)


def get_generative_answer(question, context, tokenizer, model):
    source_encoding = tokenizer(
        question,
        context,
        max_length=512,

        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt")

    generated_ids = model.generate(
        input_ids=source_encoding["input_ids"].to(device),
        attention_mask=source_encoding["attention_mask"].to(device),max_new_tokens=512,output_scores=True,
        return_dict_in_generate=True )

    preds = [tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gen_id
             in generated_ids["sequences"]]

    return "".join(preds)


def get_answer_from_model(pipe: pipeline,
                          models: dict,
                          model_dict: dict,
                          question: str,
                          context: Union[str, None]) -> str:
    """Gets an answer from a model.

    The way the pipeline object is used depends on the task being performed.

    :param pipe: Pipeline object
    :param models: Dictionary mapping models to their tokenizers and model classes
    :param model_dict: The model json which contains important information about the model
    :param question: Question
    :param context: Context
    :return: Answer
    :raise Exception if task not supported
    """
    set_seed(42)
    model_name = model_dict[MODEL_NAME]
    task = models[model_name][TASK]
    if task == QUESTION_ANSWERING:
        output = pipe(question=question, context=context)
        return output["answer"]
    elif task == TEXT2TEXT_GENERTAION:
        text = "question: {} context: {} answer: ".format(question, context)
        pattern = re.compile(r".*answer: (.*)")
        output = pipe(text, max_length=100, num_return_sequences=1)
        try:
            answer = pattern.match(output[0]["generated_text"], flags=re.DOTALL).groups()[0]
        except Exception as e:
            answer = output[0]["generated_text"]
        return answer
    elif task == TEXT_GENERATION:
        answer = get_generative_answer(question, context, models[model_name][TOKENIZER], models[model_name][MODEL])
        return answer
    elif task == TEXT_CLASSIFICATION:
        output = pipe(question)
        return output[0]["label"]
    else:
        raise Exception("Task not supported by pipeline.")
