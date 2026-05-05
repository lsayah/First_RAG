# IMPORT
from groq import Groq
from dotenv import load_dotenv
import json
import os 
import chromadb

# Import interne
from vector_db import retrieve
from sentence_transformers import SentenceTransformer


load_dotenv()


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()



def build_context(question):
	context = read_file(file_path="context.txt")

	sentence_transformer_object = SentenceTransformer("distiluse-base-multilingual-cased-v2")

	chroma = chromadb.PersistentClient(path="./my_first_vector_db")
	collection = chroma.get_or_create_collection("random_knowledge")


	chuncks = retrieve(question, sentence_transformer_object, collection, n=3)[0]

	full_context = context.replace("{{Chuncks}}", str(chuncks))
	return full_context




def answer_question(question):

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    chat_completion = client.chat.completions.create(
        messages=[

            {
                "role": "system",
                "content": build_context(question),
            },

            {
                "role": "user",
                "content": question,
            }
        ],

        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":

    response = answer_question(question="Comment s'appelle le champignon de mon gnome ? il est de couleur bleu?")
    print(response)
