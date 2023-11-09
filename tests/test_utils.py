import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

import utils


test_dataset = {"question": ["When did Beyonce start becoming popular?",
                             "In what country is Normandy located?"],
                "context": ["Beyonce Giselle Knowles-Carter (born September 4, 1981) is an American singer, songwriter"
                            ", record producer and actress. Born and raised in Houston, Texas, she performed in various"
                            " singing and dancing competitions as a child, and rose to fame in the late 1990s as lead"
                            " singer of R&B girl-group Destiny's Child.",
                            "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in"
                            " the 10th and 11th centuries gave their name to Normandy, a region in France. They were"
                            " descended from Norse  raiders and pirates from Denmark, Iceland and Norway who, under "
                            "their leader Rollo, agreed to swear fealty to King Charles III of West Francia."]}
test_json = {"model_name": "razent/SciFive-base-Pubmed_PMC",
             "type": ["extractive", "MCQ"],
             "description": "In 2019, researchers from Google released the Text-to-Text Transfer Transformer (T5) "
                            "trained on the Colossal Clean Crawled Corpus (C4) This approach achieved state-of-the-art"
                            " (SOTA) results on a diverse range of tasks related to natural language processing (NLP)."
                            " In the last decade, NLP in biomedicine has become more prominent (i.e. text mining of "
                            "scientific literature, analysis of electronic health records). This development has "
                            "created a need for NLP methods trained on corpora of biomedical literature containing "
                            "the dense technical language characteristic of scientific writing. In this report, we "
                            "introduce a T5-based model that has been successfully shifted into the biomedical domain.",
             "downloads": 5237,
             "domain": ["science"],
             "dataset": ["pubmed_qa", "zhengyun21/PMC-Patients"],
             "configs": ["pqa_labeled", ""],
             "columns": [["context", "question"], ["context", "question"]],
             "task": "question-answering",
             "split": ["train", "train"]}
description = ("2019 researchers Google released Text-to-Text Transfer Transformer T5 trained Colossal Clean Crawled "
               "Corpus C4 approach achieved state-of-the-art SOTA results diverse range tasks related natural language"
               " processing NLP last decade NLP biomedicine become prominent i.e text mining scientific literature "
               "analysis electronic health records development created need NLP methods trained corpora biomedical "
               "literature containing dense technical language characteristic scientific writing report introduce "
               "T5-based model successfully shifted biomedical domain ")
sampled_string = ("`` Beyonce Giselle Knowles-Carter born September 4 1981 American singer songwriter record producer"
                  " actress Born raised Houston Texas performed various singing dancing competitions child rose fame "
                  "late 1990s lead singer R B girl-group Destiny 's Child `` "
                  "'The Normans Norman Nourmands French Normands Latin Normanni people 10th 11th centuries gave name "
                  "Normandy region France descended Norse raiders pirates Denmark Iceland Norway leader Rollo agreed "
                  "swear fealty King Charles III West Francia 'When Beyonce start becoming popular 'In country Normandy"
                  " located")


class TestUtils(unittest.TestCase):
    @mock.patch.object(utils, "cosine_similarity")
    @mock.patch.object(utils.pd, "DataFrame")
    @mock.patch.object(utils, "TfidfVectorizer")
    def test_get_cosine_similarity_score(self, mock_vectorizer, mock_df, mock_cosine_similarity):
        vectorizer = mock.MagicMock()
        sparse_matrix = mock.MagicMock()
        term_matrix = mock.MagicMock()
        mock_vectorizer.return_value = vectorizer
        vectorizer.fit_transform.return_value = sparse_matrix
        sparse_matrix.todense.return_value = term_matrix
        mock_df.return_value = pd.DataFrame()
        mock_cosine_similarity.return_value = np.array([[0, 0.96], [0.96, 0]])
        assert utils.get_cosine_similarity_score("ans1", "ans2") == 0.96

    @mock.patch.object(utils.nltk, "word_tokenize")
    def test_get_jaccard_index(self, mock_word_tokenize):
        mock_word_tokenize.side_effect = [{"How", "are", "you", "today"}, {"Are", "you", "good", "today"}]
        assert utils.get_jaccard_index("How are you today", "Are you good today") == 1/3

    @mock.patch.object(utils.nltk, "word_tokenize")
    def test_get_jaccard_index_empty(self, mock_word_tokenize):
        mock_word_tokenize.side_effect = [{"How", "are", "you", "today"}, {}]
        assert utils.get_jaccard_index("How are you today", "") == 0

    @mock.patch.object(utils, "nlp")
    def test_get_euclidean_distance(self, mock_nlp):
        arr1 = [1, 2, 3]
        arr2 = [3, 4, 2]
        ans1_vec = np.array(arr1)
        ans2_vec = np.array(arr2)
        mock_nlp.side_effect = [mock.MagicMock(vector=ans1_vec), mock.MagicMock(vector=ans2_vec)]
        assert utils.get_euclidean_distance("How are you today", "Are you good today") == 0.049787068367863944

    def test_get_answer_similarity_score(self):
        test_function = mock.MagicMock(name='mock_a')
        with mock.patch.dict(utils.SIMILARITY_METRIC_FUNCTION_MAP, {"cosine_similarity": test_function}):
            utils.get_answer_similarity_score("ans1", "ans2")
            test_function.assert_called()

    @mock.patch.object(utils, "util")
    def test_compute_similarity_between_embeddings(self, mock_util):
        mock_util.dot_score.return_value = 0.1
        em1 = torch.Tensor([1, 2, 3])
        em2 = torch.Tensor([4, 5, 6])
        assert utils.compute_similarity_between_embeddings(em1, em2) == 0.1

    @mock.patch.object(utils, "SentenceTransformer")
    def test_fetch_embedding_model(self, mock_sent_transforner):
        mock_model = mock.MagicMock(name='multi-qa-MiniLM-L6-dot-v1')
        mock_sent_transforner.return_value = mock_model
        assert utils.fetch_embedding_model() == mock_model

    @mock.patch.object(utils, "fetch_embedding_model")
    def test_get_embeddings(self, mock_fetch_embedding_model):
        test_array = np.array([1, 2, 3])
        mock_fetch_embedding_model.return_value.encode.return_value = test_array
        assert utils.get_embeddings("text").all() == test_array.all()

    def test_sample_rows_from_dataset_column_names_exception(self):
        column_names = "question"
        with self.assertRaises(Exception) as context:
            utils.sample_rows_from_dataset("pubmed_qa", column_names)

        self.assertTrue("Column names need to be a list of column names as strings." in str(context.exception))

    @mock.patch.object(utils.pd, "DataFrame")
    @mock.patch.object(utils, "load_dataset")
    def test_sample_rows_from_dataset(self, mock_load_dataset, mock_create_dataframe):
        column_names = ("question", "context")
        mock_dataset = mock.MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.shuffle.return_value = mock_dataset
        mock_df = pd.DataFrame(test_dataset)
        assert (utils.sample_rows_from_dataset("squad", column_names)) == mock_df[list(column_names)]

    @mock.patch.object(utils, "load_dataset")
    def test_sample_rows_from_dataset_exception(self, mock_load_dataset):
        column_names = ("question", "context")
        mock_load_dataset.side_effect = Exception("ERROR!")
        with self.assertRaises(Exception) as context:
            utils.sample_rows_from_dataset("pubmed_qa", column_names)

        self.assertTrue("Error while loading dataset ERROR!" in str(context.exception))

    @mock.patch.object(utils, "load_dataset")
    def test_sample_rows_from_dataset_key_error(self, mock_load_dataset):
        column_names = ("question", "context")
        mock_dataset = mock.MagicMock()
        mock_load_dataset.return_value = mock_dataset
        mock_dataset.shuffle.return_value = mock_dataset
        df = pd.DataFrame.from_dict(test_dataset)
        df = df.drop(columns=["context"])
        with self.assertRaises(KeyError) as context:
            utils.sample_rows_from_dataset("pubmed_qa", column_names)

        self.assertTrue("None of [Index(['question', 'context'], dtype='object')] are in the [columns]" in
                        str(context.exception))

    @mock.patch.object(utils, "sample_rows_from_dataset")
    def test_get_string_to_encode(self, mock_sample_rows):
        df = pd.DataFrame.from_dict(test_dataset)
        mock_sample_rows.side_effect = [df, df]
        assert utils.get_string_to_encode(test_json) == description + sampled_string + " " + sampled_string
        assert mock_sample_rows.call_count == 2

    @mock.patch.object(utils, "get_embeddings")
    @mock.patch.object(utils, "json")
    @mock.patch.object(utils, "os")
    def test_create_map(self, mock_os, mock_json, mock_get_embeddings):
        mock_os.path.return_value.dirname.return_value = "test"
        mock_os.listdir.side_effect = [[], ["file1", "file2"]]
        mock_json.load.side_effect = [test_json, test_json]
        mock_json.dumps.return_value = '{"model": "model1"}'
        mock_get_embeddings.side_effect = [np.array([1, 2, 3]), np.array([1, 2, 3])]
        with mock.patch("builtins.open", mock.mock_open(read_data="data")) as mock_file:
            assert utils.create_map() == '{"model": "model1"}'

    @mock.patch.object(utils, "create_map")
    def test_filter_map(self, mock_create_map):
        model_map_list = [{"model": "model1", "type": ["mcq"], "downloads": 2},
                          {"model": "model2", "type": ["extractive", "mcq"], "downloads": 1},
                          {"model": "model3", "type": ["extractive"], "downloads": 1}]
        mock_create_map.return_value = json.dumps(model_map_list)
        expected_response = [{"model": "model1", "type": ["mcq"], "downloads": 2},
                             {"model": "model2", "type": ["extractive", "mcq"], "downloads": 1}]
        assert utils.filter_map("type", "mcq", 10) == expected_response

    @mock.patch.object(utils, "compute_similarity_between_embeddings")
    @mock.patch.object(utils, "get_embeddings")
    @mock.patch.object(utils, "create_map")
    def test_get_top_k_models(self, mock_create_map,
                              mock_get_embeddings,
                              mock_compute_similarity):
        model_map_list = [{"model": "model1", "embeddings": 0}, {"model": "model2", "embeddings": 0},
                          {"model": "model3", "embeddings": 0}]
        mock_create_map.return_value = json.dumps(model_map_list)
        mock_compute_similarity.side_effect = [0.1, 0.2, 0.3]
        assert utils.get_top_k_models("q", "c", 2) == [{'model': 'model3', 'embeddings': 0,
                                                        'similarity': 0.3}, {'model': 'model2', 'embeddings': 0,
                                                                             'similarity': 0.2}]

    @mock.patch.object(utils, "get_answer_similarity_score")
    def test_get_final_answer(self, mock_get_answer_similarity_score):
        test_answer_candidates = ["in the 1990s", "early 1990s", "1991"]
        test_confidence_score_of_candidates = [0.4567, 0.3356, 0.8923]

        mock_get_answer_similarity_score.side_effect = [0.2605556710562624, 0.0, 0.0]

        assert utils.get_final_answer(test_answer_candidates, test_confidence_score_of_candidates) == "in the 1990s"
