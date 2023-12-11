# Generalized Architecture for Practical Linguistic Intelligence
In the race for Artifical General Intelligence (AGI) with Large Language Models (LLMs) being the crucial advancement in language processing and question answering, we try to answer the question:  "Are singular question answering (QA) systems and LLMs capable of accurately answering a \textit{\textbf{diverse}} set of questions?" In this project, we introduce a generalized and modular architecture with a plug-and-play approach, where different specialized state-of-the-art models can be plugged in, to answer questions in their own domain. We hypothesize that this modular architecture will not compromise performance for generalizability and can thus beat unified QA models that are simultaneously trained for multiple QA tasks.

### Adding a new model to the pipeline
1. Add the model json entry to the respository folder. The entry will look as follows:
```json
{
  "model_name": <model name on HuggingFace>,
  "type": <list of model answer type: abstractive, extractive, etc.>,
  "description": <description of the model>,
  "downloads": <number of downloads on HuggingFace>,
  "dataset": <list of illustrative HuggingFace datasets>,
  "configs": <optional list of config options sometimes required by datasets>,
  "columns": <list of [question, context] pairs for each dataset>,
  "domain": <list of model domains>,
  "task": <model task (only "question-answering" as of now)>,
  "split" : <list of splits to draw example queries from in each dataset>
}
```

2. Regenerate the model map with `python utils.py`.

3. Add model entries to `load_models()` and `load_model()` in `model_pipelines.py`, including tokenizer, model, and task.