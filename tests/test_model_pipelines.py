import unittest
from unittest import mock

import model_pipelines


models = {
    "model1": {
        model_pipelines.TOKENIZER: mock.MagicMock(),
        model_pipelines.MODEL: mock.MagicMock(),
        model_pipelines.TASK: model_pipelines.QUESTION_ANSWERING
    }
}


class TestModelPipelines(unittest.TestCase):
    @mock.patch.object(model_pipelines, "pipeline")
    def test_load_pipeline(self, mock_pipeline):
        pipe = mock.MagicMock()
        mock_pipeline.return_value = pipe
        assert model_pipelines.load_pipeline(models, {"model_name": "model1"}) == pipe

    @mock.patch.object(model_pipelines, "set_seed")
    def test_get_answer_from_model_qa(self, mock_set_seed):
        pipe = mock.MagicMock()
        pipe.return_value = {"score": 0.64, "start": 276, "end": 286, "answer": "late 1990s"}
        assert model_pipelines.get_answer_from_model(pipe,
                                                     models,
                                                     {"model_name": "model1"},
                                                     "q", "c") == ("late 1990s", 0.64)

    @mock.patch.object(model_pipelines, "set_seed")
    def test_get_answer_from_model_generation(self, mock_set_seed):
        pipe = mock.MagicMock()
        pipe.return_value = [{"generated_text": "question: Q context: C answer: A"}]
        modified_models = {
            "model1": {
                model_pipelines.TOKENIZER: mock.MagicMock(),
                model_pipelines.MODEL: mock.MagicMock(),
                model_pipelines.TASK: model_pipelines.TEXT_GENERATION
            }
        }
        assert model_pipelines.get_answer_from_model(pipe,
                                                     modified_models,
                                                     {"model_name": "model1"},
                                                     "q", "c") == ("A", None)

    @mock.patch.object(model_pipelines, "set_seed")
    def test_get_answer_from_model_classification(self, mock_set_seed):
        pipe = mock.MagicMock()
        pipe.return_value = [{'label': 'bearish', 'score': 0.94}]
        modified_models = {
            "model1": {
                model_pipelines.TOKENIZER: mock.MagicMock(),
                model_pipelines.MODEL: mock.MagicMock(),
                model_pipelines.TASK: model_pipelines.TEXT_CLASSIFICATION
            }
        }
        assert model_pipelines.get_answer_from_model(pipe,
                                                     modified_models,
                                                     {"model_name": "model1"},
                                                     "q", "c") == ("bearish", 0.94)

    @mock.patch.object(model_pipelines, "set_seed")
    def test_get_answer_from_model_exception(self, mock_set_seed):
        pipe = mock.MagicMock()
        pipe.return_value = [{'label': 'bearish', 'score': 0.94}]
        modified_models = {
            "model1": {
                model_pipelines.TOKENIZER: mock.MagicMock(),
                model_pipelines.MODEL: mock.MagicMock(),
                model_pipelines.TASK: "unknown_task"
            }
        }
        with self.assertRaises(Exception) as context:
            model_pipelines.get_answer_from_model(pipe,
                                                  modified_models,
                                                  {"model_name": "model1"},
                                                  "q", "c")

        self.assertTrue("Task not supported by pipeline." in str(context.exception))
