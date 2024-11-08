# Colab Link: https://colab.research.google.com/drive/1SL7-0w-2amTktgj5eQEg4Pcf2mt_DsIU?usp=sharing

import os
import sys

from autogen import AssistantAgent

sys.path.append(os.path.abspath("/workspaces/autogen/autogen/agentchat/contrib"))

from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Accepted file formats for that can be stored in
# a vector database instance
from autogen.retrieve_utils import TEXT_FORMATS

OPEN_API_KEY = ""

config_list = [{"model": "gpt-4o-mini", "api_key": OPEN_API_KEY, "api_type": "openai"}]
assert len(config_list) > 0
print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])


print("Accepted file formats for `docs_path`:")
print(TEXT_FORMATS)


# 1. create an AssistantAgent instance named "assistant"
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

config_list[0]["model"] = "gpt-35-turbo"  # change model to gpt-35-turbo


corpus_file = "https://huggingface.co/datasets/thinkall/NaturalQuestionsQA/resolve/main/corpus.txt"

# Create a new collection for NaturalQuestions dataset
# `task` indicates the kind of task we're working on. In this example, it's a `qa` task.
# ragproxyagent = RetrieveUserProxyAgent(
#     name="ragproxyagent",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
#     retrieve_config={
#         "task": "qa",
#         "docs_path": corpus_file,
#         "chunk_token_size": 2000,
#         "model": config_list[0]["model"],
#         "client": chromadb.PersistentClient(path="/tmp/chromadb"),
#         "collection_name": "natural-questions",
#         "chunk_mode": "one_line",
#         "embedding_model": "all-MiniLM-L6-v2",
#     },
# )
ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "qa",
        "docs_path": corpus_file,
        "chunk_token_size": 2000,
        "model": config_list[0]["model"],
        "vector_db": "couchbase",  # Couchbase Capella VectorDB
        "collection_name": "demo_collection",  # Couchbase Capella collection name to be utilized/created
        "db_config": {
            "connection_string": "",  # Couchbase Capella connection string
            "username": "",  # Couchbase Capella username
            "password": "",  # Couchbase Capella password
            "bucket_name": "autogen",  # Couchbase Capella bucket name
            "scope_name": "autogenscope",  # Couchbase Capella scope name
            "index_name": "autogen",  # Couchbase Capella index name to be created
        },
        "get_or_create": True,  # set to False if you don't want to reuse an existing collection
        "overwrite": False,  # set to True if you want to overwrite an existing collection, each overwrite will force a index creation and reupload of documents
       "chunk_mode": "one_line",
        "embedding_model": "all-MiniLM-L6-v2",
    },
    code_execution_config=False,  # set to False if you don't want to execute the code
)



import json
# queries_file = "https://huggingface.co/datasets/thinkall/NaturalQuestionsQA/resolve/main/queries.jsonl"
queries = """{"_id": "ce2342e1feb4e119cb273c05356b33309d38fa132a1cbeac2368a337e38419b8", "text": "what is non controlling interest on balance sheet", "metadata": {"answer": ["the portion of a subsidiary corporation 's stock that is not owned by the parent corporation"]}}
{"_id": "3a10ff0e520530c0aa33b2c7e8d989d78a8cd5d699201fc4b13d3845010994ee", "text": "how many episodes are in chicago fire season 4", "metadata": {"answer": ["23"]}}
{"_id": "fcdb6b11969d5d3b900806f52e3d435e615c333405a1ff8247183e8db6246040", "text": "what are bulls used for on a farm", "metadata": {"answer": ["breeding", "as work oxen", "slaughtered for meat"]}}
{"_id": "26c3b53ec44533bbdeeccffa32e094cfea0cc2a78c9f6a6c7a008ada1ad0792e", "text": "has been honoured with the wisden leading cricketer in the world award for 2016", "metadata": {"answer": ["Virat Kohli"]}}
{"_id": "0868d0964c719a52cbcfb116971b0152123dad908ac4e0a01bc138f16a907ab3", "text": "who carried the usa flag in opening ceremony", "metadata": {"answer": ["Erin Hamlin"]}}
"""
queries = [json.loads(line) for line in queries.split("\n") if line]
questions = [q["text"] for q in queries]
answers = [q["metadata"]["answer"] for q in queries]
print(questions)
print(answers)



for i in range(len(questions)):
    print(f"\n\n>>>>>>>>>>>>  Below are outputs of Case {i+1}  <<<<<<<<<<<<\n\n")

    # reset the assistant. Always reset the assistant before starting a new conversation.
    assistant.reset()

    qa_problem = questions[i]
    chat_result = ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=qa_problem, n_results=30
    )
