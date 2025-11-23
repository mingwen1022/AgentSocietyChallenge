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
from memory_modules_custom import *

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


def load_first_task(dataset="goodreads"):
    """
    dataset: 'goodreads' or 'amazon' or 'yelp'
    Returns: the parsed task dict
    """

    base_path = f"./example/track2/{dataset}/tasks"

    task_files = sorted(
        [
            f
            for f in os.listdir(base_path)
            if f.startswith("task_") and f.endswith(".json")
        ]
    )

    if not task_files:
        raise FileNotFoundError(f"No task files found in {base_path}")

    # read_first_file (task_0.json)
    first_file = os.path.join(base_path, task_files[0])
    print(f"Loading task file: {first_file}")

    with open(first_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    return task


def load_first_groundtruth(dataset="goodreads"):
    """
    dataset: 'goodreads' or 'amazon' or 'yelp'
    Returns: the parsed task dict
    """

    base_path = f"./example/track2/{dataset}/groundtruth"

    groundtruth_files = sorted(
        [
            f
            for f in os.listdir(base_path)
            if f.startswith("groundtruth_") and f.endswith(".json")
        ]
    )

    if not groundtruth_files:
        raise FileNotFoundError(f"No task files found in {base_path}")

    # read first groundtruth(groundtruth_0.json)
    first_file = os.path.join(base_path, groundtruth_files[0])
    print(f"Loading task file: {first_file}")

    with open(first_file, "r", encoding="utf-8") as f:
        task = json.load(f)

    return task


class TestRecommendationAgent(RecommendationAgent):
    """
    Only for local testing of planning module, without Simulator.
    """

    def __init__(self, llm: LLMBase, dataset: str = "goodreads"):
        super().__init__(llm=llm)
        self.dataset = dataset
        self.planning = PlanningIOCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)

    def workflow(self):

        print("\n===== TEST: Planning Module (Recommendation Voyager) =====\n")
        # load task from example/track2/goodreads/tasks/task_0.json
        test_task = load_first_task(dataset="goodreads")
        task_description = json.dumps(test_task, indent=2)

        memory = self.memory()

        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot="",
        )
        for step in plan:
            print(f"Step: {step['description']}")
            print(f"Reasoning Instruction: {step['reasoning instruction']}")
            print("----")

        # simulated reasoning and tooluse results
        tooluse_result = "example_tooluse: simulated tool use (fetch user/item/review)"
        reasoning_result = ["cand_1", "cand_5", "cand_3", "cand_2", "cand_9"]
        # end of reasoning and tooluse

        # eval groundtruth
        groundtruth = "cand_3"
        top3 = reasoning_result[:3]
        is_correct = groundtruth in top3

        # Add Memory
        trajectory = (
            f"Task:\n"
            f"    {task_description}\n\n"
            f"Plan:\n"
            f"    {plan}\n\n"
            f"Reasoning:\n"
            f"    {reasoning_result}\n\n"
            f"ToolUse:\n"
            f"    {tooluse_result}\n\n"
            f"Is Correct:\n"
            f"    {is_correct}\n"
        )

        print("\n[Trajectory to Memory]:", trajectory)

        self.memory("review: " + trajectory)
        print("[Memory Added]\n")

        return reasoning_result


if __name__ == "__main__":
    # api_key_gemini = GEMINI_KEY
    # api_key_openai = OPENAI_API_KEY
    api_key_google = GOOGLE_API_KEY
    # llm_gemini = GeminiLLM(api_key=api_key_gemini, model="gemini-1.5-pro")
    # llm_openai = OpenAILLM(api_key=api_key_openai, model="gpt-4o")
    llm_google = GeminiLLM(api_key=api_key_google, model="gemini-2.5-flash")
    print("\n===== TEST: Planning Module (Recommendation Voyager, Class Mode) =====\n")
    agent = TestRecommendationAgent(llm_google, dataset="goodreads")
    plan = agent.workflow()
