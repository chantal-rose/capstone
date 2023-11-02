import unittest
from unittest import mock

import utils


class TestUtils(unittest.TestCase):
    @mock.patch.object(utils, "get_answer_similarity_score")
    def test_get_final_answer(self, mock_get_answer_similarity_score):
        test_answer_candidates = ["in the 1990s", "early 1990s", "1991"]
        test_confidence_score_of_candidates = [0.4567, 0.3356, 0.8923]

        mock_get_answer_similarity_score.side_effect = [0.2605556710562624, 0.0, 0.0]

        assert utils.get_final_answer(test_answer_candidates, test_confidence_score_of_candidates) == "in the 1990s"
