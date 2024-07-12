import pymongo
from decouple import config

host = config("DATABASE_HOST")
username = config("DATABASE_USERNAME")
password = config("DATABASE_PASSWORD")
database = config("DATABASE_NAME")

def connect():
    client = pymongo.MongoClient(host = host, username = username, password = password)
    return client[database]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
