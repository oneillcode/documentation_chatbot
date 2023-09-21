import os
from dotenv import load_dotenv
import pandas as pd
import openai
import numpy as np
import pinecone


class Chatbot:
    # Class Init
    def __init__(self,
                 embedding_model="text-embedding-ada-002",
                 gpt_model="gpt-3.5-turbo",
                 debug=False):
        self.embedding_model = embedding_model
        self.gpt_model = gpt_model
        self.debug = debug

        load_dotenv()

        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
                      environment=os.getenv("PINECONE_ENV"))

        # Get API key
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def create_pinecone_context(self, query, max_len=1800):

        try:
            # Get the embeddings for the question
            embedding = openai.Embedding.create(
                input=query,
                engine=self.embedding_model
            )['data'][0]['embedding']

            # Get Pincecone index - this should be the index where vectors are stored from web scraper
            index = pinecone.Index('starburst')

            vector_response = index.query(vector=embedding,
                                          top_k=5,
                                          include_metadata=True,
                                          max_len=max_len
                                          )['matches']

            # Check if vector response is empty
            if len(vector_response) == 0:
                return "I do not know this answer."

            context = ""

            # Loop through vector response and add to context
            for i in range(len(vector_response)):
                context += vector_response[i]['metadata']['text'] + " "

            return context
        except Exception as e:
            print(e)
            return "I do not know this answer and have no context of it."

    def create_message(self, query):
        context = self.create_pinecone_context(
            query,
            max_len=1000,
        )

        if context == "I do not know this answer.":
            return context

        # If debug, print the raw model response
        if self.debug:
            print("Context:\n" + context)
            print("\n\n")

        header = "Answer the questions in the conversation using the context below. You are an expert on Starburst Data and a friendly assistant. Do not instruct the user to read to docs but instead, explain the answers in a concise manner. If the answer is not known, 'I don't know' is an appropriate response. If writing code or YAML files, surround it with ``` CODE HERE ```.\n\n "
        question = "Question: How would you answer the below question as if you were an instructor at Starburst data? \n"

        return header + context + question + query

    def answer_question(
        self,
        question="",
    ):

        try:
            # Create a completions using the questin and context
            message = self.create_message(query=question)

            if message == "I do not know this answer.":
                return message

            messages = [
                {"role": "system", "content": "You help Starburst Data users understand the product. You answer questions about Starburst Data and Trino. Your tone is helpful and joyful."},
                {"role": "user", "content": message},
            ]

            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=messages,
                temperature=0,
                stream=True
            )

            print("\n[Chatbot]: ", end="")
            for chunk in response:
                print(chunk["choices"][0]['delta']['content'], end="")

        except Exception as e:
            print(e)
            return ""

    def start(self):
        input_val = input(
            '[Chatbot]: Hello! How can I help you with Starburst Data? Type "exit" to close chat.\n\n> ')

        while input_val != "exit":
            self.answer_question(question=input_val)
            input_val = input('\n\n> ')


chatbot = Chatbot()
chatbot.start()
