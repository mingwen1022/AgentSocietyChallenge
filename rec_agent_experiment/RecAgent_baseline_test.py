import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from gemini import GeminiLLM
import re
import logging
import time
from dotenv import load_dotenv
from planning_module_custom import *

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

logging.basicConfig(level=logging.INFO)


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        a = len(encoding.encode(string))
    except:
        print(encoding.encode(string))
    return a


# llm api connection test only
def test_llm():
    api_key_gemini = GEMINI_KEY
    api_key_openai = OPENAI_API_KEY
    api_key_google = GOOGLE_API_KEY
    llm_gemini = GeminiLLM(api_key=api_key_gemini, model="gemini-1.5-pro")
    llm_openai = OpenAILLM(api_key=api_key_openai, model="gpt-4o")
    llm_google = GeminiLLM(api_key=api_key_google, model="gemini-2.5-flash")

    result = llm_google(
        messages=[{"role": "user", "content": "Say hello briefly."}],
        temperature=0.1,
        max_tokens=50,
    )
    print("Gemini output:")
    print(result)


def test_planning_module():
    api_key_gemini = GEMINI_KEY
    api_key_openai = OPENAI_API_KEY
    api_key_google = GOOGLE_API_KEY
    llm_gemini = GeminiLLM(api_key=api_key_gemini, model="gemini-1.5-pro")
    llm_openai = OpenAILLM(api_key=api_key_openai, model="gpt-4o")
    llm_google = GeminiLLM(api_key=api_key_google, model="gemini-2.5-flash")

    print("\n===== TEST: Planning Module (Recommendation Voyager) =====\n")

    test_task = {
        "type": "recommendation",
        "user_id": "ztgVL0NPadoUwCO9MWeUUQ",
        "candidate_category": "business - Shopping",
        "candidate_list": [
            "OjFr_sk32NOhYSvA_Ucd5Q",
            "PdBwl7tlFhOBR3p4kMd9Hw",
            "FbY5HjT_nCqfB8PiXX-fcQ",
            "oFu61fiwKh6W_zgGjATfyw",
            "QOUqT4PuH2Xm-ky0R87JNg",
            "BrO4rhvgGU2vXx4cvVJIpg",
            "uFs6biPJw2FlVY3taR4QNQ",
            "pRYs_U3tiTisUazOzDgLaA",
            "qF1NTfE0yfbTc1kb2mX1FA",
            "0d7nPS5dv42stQqdZbh08g",
            "WNFxt2TyMDEuNNNZ9Z9zRQ",
            "g-cQ3TeR7lcY_TObcDkw6w",
            "IaelRCI4Ah5oxiHuBnFC7w",
            "9kfDDNapNWKF5B41X27HkA",
            "4hmaKbOARJsP_8UfRssQJg",
            "ClIkpkKO-Es8MHlsfDOKMQ",
            "OQqBFuA5tcxdHog8YgMRcQ",
            "ZVrOGpZe5usRbdxxtmxHoQ",
            "o2oD8bGW3oMaaTeSOqi-Kg",
            "BzDbphyIfHWZUoaQQdAUYw",
        ],
        "loc": [4621600.795281307, 5685269.728156481],
    }

    # test_task_safe = {
    #     "type": "recommendation",
    #     "user_id": test_task["user_id"],
    #     "candidate_count": len(test_task["candidate_list"]),
    #     "candidate_category": test_task["candidate_category"],
    #     "has_location": True,
    # }

    # create task_description
    task_description = json.dumps(test_task, indent=2)

    # create planner,
    # feedback: for planning validation(validate whether the output is in correct structure or contain required elements. Not yet implemented)
    # few_shot: param for long-term memory(successful examples)
    planner = PlanningIOCustom(llm_google)
    plan = planner(
        task_type="Recommendation Task",
        task_description=task_description,
        feedback="",
        few_shot="",
    )
    for step in plan:
        print(f"Step: {step['description']}")
        print(f"Reasoning Instruction: {step['reasoning instruction']}")
        print("----")


if __name__ == "__main__":
    test_planning_module()


# output_openai:
# [
#   {
#     "description": "Retrieve user profile data for user_id 'ztgVL0NPadoUwCO9MWeUUQ' from the user dataset.",
#     "reasoning instruction": "Understand the user's preferences and behavior by analyzing their review count, average stars, and any compliments they have received."
#   },
#   {
#     "description": "Retrieve metadata for each item in the candidate_list from the item dataset.",
#     "reasoning instruction": "Gather information about each business, including their name, average stars, categories, and review count, to assess their general popularity and relevance."
#   },
#   {
#     "description": "Retrieve all reviews written by user_id 'ztgVL0NPadoUwCO9MWeUUQ' from the review dataset.",
#     "reasoning instruction": "Analyze the user's past reviews to identify patterns in their ratings and preferences, which can help predict their future interests."
#   },
#   {
#     "description": "Retrieve all reviews for each item in the candidate_list from the review dataset.",
#     "reasoning instruction": "Examine the reviews for each candidate item to understand their strengths and weaknesses from the perspective of other users, which can aid in determining their suitability for the current user."
#   },
#   {
#     "description": "Filter candidate items based on proximity to the user's location (latitude: 4621600.795281307, longitude: 5685269.728156481).",
#     "reasoning instruction": "Consider the geographical proximity of each business to the user, as closer businesses may be more convenient and thus more relevant."
#   },
#   {
#     "description": "Rank the candidate items based on relevance to the user, considering user preferences, item popularity, review sentiments, and location proximity.",
#     "reasoning instruction": "Integrate all gathered data to prioritize the candidate items, ensuring the most relevant and appealing options are recommended to the user."
#   }
# ]


# output_gemini
# [
#     {
#         "step": 1,
#         "description": "Retrieve the user's profile details and all reviews submitted by the user.",
#         "reasoning": (
#             "Analyze the user's historical reviewing behavior, sentiment tendencies, "
#             "and the categories/business types they frequently interact with. "
#             "This helps establish their overall preferences and dislikes."
#         )
#     },
#     {
#         "step": 2,
#         "description": (
#             "For each item in `candidate_list`, retrieve detailed metadata including "
#             "name, categories, star rating, review count, and location."
#         ),
#         "reasoning": (
#             "Provides essential item attributes, which are necessary for aligning "
#             "the candidates with user preferences and computing geographic proximity."
#         )
#     },
#     {
#         "step": 3,
#         "description": (
#             "Process the user's past review logs to identify commonly reviewed categories, "
#             "their average ratings for those categories, and sentiment trends—particularly "
#             "focusing on 'Shopping' or related domains."
#         ),
#         "reasoning": (
#             "Determines which business types the user typically likes or dislikes and "
#             "reveals preferences specifically related to shopping behaviors."
#         )
#     },
#     {
#         "step": 4,
#         "description": (
#             "For each candidate item, retrieve a sample of its reviews and examine the "
#             "average star rating and total number of reviews."
#         ),
#         "reasoning": (
#             "Helps understand public perception, extract key pros/cons, and measure "
#             "overall popularity and external validation for each candidate."
#         )
#     },
#     {
#         "step": 5,
#         "description": (
#             "Using the user's provided `loc` (latitude, longitude) and each candidate "
#             "item's location, compute geographic distance."
#         ),
#         "reasoning": (
#             "Distance is an important factor in local recommendations. "
#             "Closer items generally provide better convenience and relevance."
#         )
#     }
# ]
