import openai
import re


openai.api_key = ""
content = ""
domain_pattern = re.compile(r"^Domain:\s([\w \d]+)")
type_pattern = re.compile(r"^Type:\s([\w \d]+)")
question_pattern = re.compile(r"^Question:\s([\w \d]+)")
context_pattern = re.compile(r"^Context:\s([\w \d]+)")


class GPT4InputParser:  # pragma: no cover
    def __init__(self):
        self.model = "gpt-4"
        self.type = ""
        self.domain = ""
        self.question = ""
        self.context = ""

    def parse(self, prompt):
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": content},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0)
        output = response['choices'][0]['message']['content']
        try:
            self.domain = domain_pattern.match(output).groups()[0]
            self.type = type_pattern.match(output).groups()[0]
            self.question = question_pattern.match(output).groups()[0]
            self.context = context_pattern.match(output).groups()[0]
        except Exception as e:
            print("Error while parsing the user input by GPT-4, {}".format(e))
