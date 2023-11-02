from preprocess import Ranker
from models.question_answering_models import ExtractiveQuestionAnsweringModel, Text2TextModel, InstructModel
from utils import filter_map, get_final_answer, create_map
import sys

model_classes = {
        'Text2TextModel': Text2TextModel,
        'InstructModel':InstructModel,
        'ExtractiveQuestionAnsweringModel':ExtractiveQuestionAnsweringModel
    }
tokenizer = ''


class Main:
    def __init__(self, parameters):
        self.model_classes = {
            'Text2TextModel': Text2TextModel,
            'InstructModel': InstructModel,
            'ExtractiveQuestionAnsweringModel': ExtractiveQuestionAnsweringModel
        }
        self.tokenizer = ''

        self.query = parameters[0]
        self.context = parameters[1]
        self.k = parameters[2]
        self.domain = None
        self.type = None
        if len(parameters) > 3:
            self.domain = parameters[3]
            self.type = parameters[4]
        self.output = ''

    def parse_input(self):
        if not self.domain:
            self.domain = ' '  # call gpt4
        if not self.type:
            self.type = ' '  # call gpt4

    def get_top_models(self):
        # embedding based model search
        ranker = Ranker()
        top_k_embedding_models = ranker.get_top_k_models(self.query, self.context, self.k)

        # domain based model search
        top_k_domain_models = filter_map("domain", self.domain, self.k)

        # type based model search
        top_k_type_models = filter_map("type", self.type, self.k)

        return [top_k_embedding_models, top_k_domain_models, top_k_type_models]

    def get_model_outputs(self, top_models):

        answer_candidates = []
        confidence_score_of_candidates = []

        for pipeline in top_models:
            for model in pipeline:
                task = model['task']
                inference_class = model_classes[model['inference_class']](model['model_name'], tokenizer)

                output = inference_class.predict(self.query, self.context)
                answer_candidates.append(output['answer'])
                confidence_score_of_candidates.append(output['score'])
        return answer_candidates, confidence_score_of_candidates

    def run(self):
        # create map
        create_map(True, [])

        self.parse_input()
        top_models = self.get_top_models()

        top_answer_candidates, top_answer_candidates_confidence_scores = self.get_model_outputs(top_models)

        gen_qa_output = get_final_answer(top_answer_candidates, top_answer_candidates_confidence_scores)

        self.output = gen_qa_output

    def verify(self):

        if self.output == '': #if no answer was generate
            return False

        # reformulate for boolqa
        # verify_query = verify_reformulation(overall_output, query, context)

        # pass through boolqa
        pass



if __name__ == "__main__":

    gen_qa = Main(sys.argv)

    run_complete = False

    attempts = 0
    while not run_complete:
        attempts += 1
        gen_qa.run()
        output_verified = gen_qa.verify()

        if output_verified:
            run_complete = True
        elif attempts > 5:
            break
        else:
            continue
            #reformulate query
            #reset params

    if run_complete and len(gen_qa.output) > 0:
        print(gen_qa.output)
    else:
        print('Sorry the system could not find a response!')









