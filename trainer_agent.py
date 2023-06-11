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

    def generate_training_action(self, history, hyperparams: dict, thought_message = None, model: str = "gpt-3.5-turbo", temperature: float = 0.1, presence_penalty: float = 0.0, frequency_penalty: float = 0.0, max_tokens: int = 1000):
        self.add_message("You are an AI model specialized in neural network training, you respond only in the format requested and may only take the actions provided.", role="system")

        formatting = ("""
        Here is the history of the neural network training, note that all of the changes in model hyperparams were made by you:
        """+
        str(history)
        +"""

        You are provided these actions:
        # "Change" - changes the model hyperparameters that is being trained on the MNIST data
        ## Specified Response: hyperparams = {"num_filters": list with length>=1, "kernel_size": int>=1, "hidden_sizes": list with length>=1, "learning_rate": float<1.0, "epochs": int>=1, "batch_size": int,"lin_dropout": float<1.0, "conv_dropout": float<1.0}
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
        Action: Think
        Response: I am analyzing the above history and have identified that the model performs better when the hyper parameters are ...
        ```
        You should focus on hyperparameters other than epoch to start out. Try to find the best set of Convulutional Layers, Hidden Layers, etc. before moving on to longer training runs. Please analyze the above history and metrics and respond in the specified format.

        You may only think ONCE, do not generate Think action twice.
        """)
        formatting = '\n'.join([line.strip() for line in formatting.split('\n')])

        self.add_message(formatting, role="user")

        if thought_message != None:
            self.add_message(thought_message, role="assistant")

        """
        print("-----------------------")
        for message in self.messages:
            print(message['content'])
        print("-----------------------")
        """

        max_attempts = 3  # Maximum number of attempts
        current_attempt = 1  # Current attempt count

        while current_attempt <= max_attempts:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.messages,
                temperature=temperature,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens
            )

            if response['choices'][0]['message']['content']:
                # If the response is not empty, break out of the loop
                break

            # Adjust the parameters for the next attempt
            temperature -= 0.1
            presence_penalty -= 0.1
            frequency_penalty -= 0.1
            max_tokens += 10

            current_attempt += 1

        self.clear_messages()

        return response['choices'][0]['message']['content']