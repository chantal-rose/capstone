import unittest
from unittest import mock


class TestMain(unittest.TestCase):
    @mock.patch("main.get_final_answer")
    @mock.patch("main.get_answer_from_model")
    @mock.patch("main.load_pipeline")
    @mock.patch("main.filter_map")
    @mock.patch("main.get_top_k_models")
    @mock.patch("main.GPT4InputParser")
    @mock.patch("main.load_models")
    def test_send_input_to_system(self, mock_load_models,
                                  mock_parser,
                                  mock_get_top_k_models,
                                  mock_filter_map,
                                  mock_load_pipeline,
                                  mock_get_answer_from_model,
                                  mock_get_final_answer):
        from main import send_input_to_system

        mock_load_models.return_value = None
        question = ("Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic "
                    "rhinosinusitis with nasal polyps or eosinophilia?")
        context = ("Chronic rhinosinusitis (CRS) is a heterogeneous disease with an uncertain pathogenesis. "
                   "Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population which has"
                   " been implicated in driving Th2 inflammation in CRS; however, their relationship with clinical"
                   " disease characteristics has yet to be investigated.")
        parser = mock.MagicMock(type="bio", domain="extractive", question=question, context=context)
        mock_parser.return_value.parse.return_value = parser
        models = [{"model_name": "model_name"}]
        mock_get_top_k_models.return_value = models
        mock_filter_map.side_effect = [models, models]
        mock_get_answer_from_model.side_effect = [("answer1", 0.4),
                                                  ("answer2", None),
                                                  ("answer3", 0.6)]
        mock_get_final_answer.return_value = "answer3"
        send_input_to_system(models[0], "{} context: {}".format(question, context))
        assert mock_load_pipeline.call_count == 3
        assert mock_get_answer_from_model.call_count == 3
        mock_get_final_answer.assert_called_with(["answer1", "answer2", "answer3"], [0.4, 0.5, 0.6])
