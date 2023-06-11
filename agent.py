from config import OPENAI_API_SECRET_KEY
import openai


openai.api_key = OPENAI_API_SECRET_KEY

class Agent:
    def __init__(self, system = "You are a helpful assistant", additional_messages: list = None):
        self.system = system
        self.history = []
        if additional_messages is not None:
            for message in additional_messages:
                self.add_message(message)

    def add_message(self, message: str, role: str = "user"):
        message = {"role": role, "content": message}
        self.history.append(message)

    def clear_history(self):
        self.history = []

    def generate_response(self, message: str = None, history: bool = True, model = "gpt-4", temperature = 0.5, presence_penalty = 0.1, frequency_penalty = 0.1, max_tokens = 1000):
        messages = [{"role": "system", "content": self.system}]

        #adding a user message from params to history
        if message is not None:
            self.history.append({"role": "user", "content": message})

        messages.extend(self.history)

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature = temperature,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            max_tokens = max_tokens
        )


        #adding response of agent to history
        self.history.append({"role": response['choices'][0]['message']['role'], "content":response['choices'][0]['message']['content']})

        if not history: self.clear_history()

        return response['choices'][0]['message']['content']