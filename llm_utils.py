import re
import os
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from skllm import ZeroShotGPTClassifier
from skllm import MultiLabelZeroShotGPTClassifier
from skllm.config import SKLLMConfig
import pandas as pd

load_dotenv()
client = OpenAI()

content = ""
context_pattern = re.compile(r"^Context:\s([\w ]+)")
domain_pattern = re.compile(r"^Domain:\s([\w ]+)")
question_pattern = re.compile(r"^Question:\s([\w ]+)")
type_pattern = re.compile(r"^Type:\s([\w ]+)")

DEFAULT_MODEL = "gpt-3.5-turbo-1106"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm(messages: list[dict[str, str]], model: str = DEFAULT_MODEL) -> str:
    """Interacts with an OpenAI model and returns a response,

    :param messages: Messages with context to send to LLM
    :param model: The model to use
    :return Response
    """
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    response = completion.choices[0].message.content
    return response

def domain_label(context):
    SKLLMConfig.set_openai_key(os.environ["OPENAI_API_KEY"])
    SKLLMConfig.set_openai_org(os.environ["OPENAI_ORG_KEY"])
    d = {"generic-QA":"nan","legal-QA":"legal",'biology-QA':'bio'}
    clf = ZeroShotGPTClassifier(openai_model=DEFAULT_MODEL)
    clf.fit(None, ["biology-QA","legal-QA","generic-QA"])
    predicted = clf.predict([context[:3000]])
    return d[predicted[0]]


class GPT4InputParser:  # pragma: no cover
    def __init__(self):
        self.model = "gpt-4"
        self.type = ""
        self.domain = ""
        self.question = ""
        self.context = ""

    def parse(self, prompt):
        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]
        output = query_llm(messages, self.model)
        try:
            self.domain = domain_pattern.match(output).groups()[0]
            self.type = type_pattern.match(output).groups()[0]
            self.question = question_pattern.match(output).groups()[0]
            self.context = context_pattern.match(output).groups()[0]
        except Exception as e:
            print("Error while parsing the user input by GPT-4, {}".format(e))
