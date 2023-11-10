from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()
client = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm(messages: list[str]) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    topic = completion.choices[0].message.content
    return topic