from config import OPENAI_API_SECRET_KEY
import openai
openai.api_key = OPENAI_API_SECRET_KEY
from typing import List

class TrainerAgent():
    def __init__(self):
        self.messages = []

    def add_message(self, message: str, role: str = "user"):
        message = {"role": role, "content": message}
        self.messages.append(message)

    def clear_messages(self):
        self.messages = []

    def generate_training_action(self, history: List[dict], hyperparams: dict, thought_message = None, model: str = "gpt-4", temperature: float = 0.2, presence_penalty: float = 0.1, frequency_penalty: float = 0.1, max_tokens: int = 1000):
        self.add_message("You are an AI model specialized in neural network training, you respond only in the format requested and may only take the actions provided.", role="system")
        self.add_message("Here is the history and hyperparams of the neural network training:", role="user")
        self.add_message(str(history), role="user")
        self.add_message(str(hyperparams), role="user")

        formatting = ("""
        You are provided these actions:
        # "Rerun" - reruns the model with the same hyperparameters that is being trained on the MNIST data
        ## Specified Response: hyperparams = {"num_filters": list len >= 1, "kernel_size": int>=1, "hidden_sizes": list len >= 1, "learning_rate": float<1.0, "epochs": int>=1, "batch_size": int,"lin_dropout": float<1.0, "conv_dropout": float<1.0}
        
        # "Change" - changes the model hyperparameters that is being trained on the MNIST data
        ## Specified Response: RUN
        
        # "Think" - allows you to think and write about the changes to be made next. This can only be done ONCE
        ## Specified Response: any text or thoughts you would like to generate, but do NOT generate another action


        You must respond in the following format:
        ```
        Action: action
        Response: reponse format specified by action
        ```

        Here are example responses:
        ```
        Action: Change
        Response: hyperparams = {"num_filters": [16], "kernel_size": 3, "hidden_sizes": [64], "learning_rate": 0.001, "epochs": 2, "batch_size": 100, "lin_dropout": 0.5, "conv_dropout": 0.25}
        ```
        ```
        Action: Rerun
        Response: RUN
        ```
        ```
        Action: Think
        Response: I am analyzing the above history and have identified that the model performs better when the hyper parameters are ...
        ```

        """)
        formatting = '\n'.join([line.strip() for line in formatting.split('\n')])

        self.add_message(formatting, role="user")

        self.add_message("Please analyze the above history and metrics and respond in the specified format.", role="user")
        
        """
        print("-----------------------")
        for message in self.messages:
            print(message['content'])
        print("-----------------------")
        """

        if thought_message != None:
            self.add_message(thought_message, role="assistant")

        response = openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            temperature = temperature,
            presence_penalty = presence_penalty,
            frequency_penalty = frequency_penalty,
            max_tokens = max_tokens
        )

        self.clear_messages()

        return response['choices'][0]['message']['content']