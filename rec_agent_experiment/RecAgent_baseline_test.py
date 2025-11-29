import sys, os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM, OpenAILLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import (
    ReasoningBase,
    ReasoningIO,
)
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
from websocietysimulator.tools.interaction_tool import InteractionTool
from websocietysimulator.agent.modules.pairwise_modules import PairwiseRanker
from gemini import GeminiLLM
import re
import logging
import time
from dotenv import load_dotenv
from websocietysimulator.agent.modules.planning_module_custom import *
from websocietysimulator.agent.modules.memory_modules_custom import *

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
        self.planning = PlanningVoyagerCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm, reset=False)
        self.reasoning = ReasoningIO(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        # Local interaction tool and info orchestrator for profile pipeline testing
        # Note: current data_dir uses Yelp-style processed data in ./data
        self.interaction_tool = InteractionTool(data_dir="./data")
        self.info_orchestrator = InfoOrchestrator(
            memory=self.memory,
            llm=self.llm,
            schema_fitter=None,
            interaction_tool=None,
            use_fixed_item_params=True,
            max_candidates_to_profile=None,
        )
        self._schema_fitter_llm = self.llm
        # Pairwise reranker on top-K items
        self.pairwise_ranker = PairwiseRanker(self.llm)
        # store last run summary (for multi-task experiments)
        self.last_result_summary = None

    def workflow(self, test_task=None, groundtruth_item=None, task_name=None):

        print("\n===== TEST: Planning Module (Recommendation Voyager) =====\n")
        # load task and groundtruth from example/track2/{dataset} if not provided
        if test_task is None:
            test_task = load_first_task(dataset=self.dataset)
        if groundtruth_item is None:
            gt = load_first_groundtruth(dataset=self.dataset)
            groundtruth_item = gt.get("ground truth")

        task_description = json.dumps(test_task, indent=2)

        # Retrieve one long-term memory trajectory as few-shot guidance for planning
        task_query = json.dumps(test_task, ensure_ascii=False)
        few_shot = self.memory(task_query) or ""

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

        # ----- InfoOrchestrator: build user/item profiles based on planner steps -----
        # Initialize SchemaFitterIO once we have an interaction_tool
        if self.info_orchestrator and self.interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(
                    self._schema_fitter_llm, self.interaction_tool
                )
                self.info_orchestrator.schema_fitter = schema_fitter
            self.info_orchestrator.interaction_tool = self.interaction_tool

        profiles = self.info_orchestrator(
            planner_steps=plan,
            user_id=test_task["user_id"],
            candidate_list=test_task["candidate_list"],
        )

        user_profile = profiles.get("user_profile")
        item_profiles = profiles.get("item_profiles", [])

        print("\n===== INFO-ORCHESTRATOR OUTPUT =====")
        if user_profile:
            print("\n[User profile]")
            print(json.dumps(user_profile, indent=2, ensure_ascii=False))
        else:
            print("\n[User profile] None")

        if item_profiles:
            print(f"\n[Item profiles] total={len(item_profiles)}, showing first 3:")
            for p in item_profiles[:3]:
                print(json.dumps(p, indent=2, ensure_ascii=False))
        else:
            print("\n[Item profiles] None")
        print("===== END INFO-ORCHESTRATOR OUTPUT =====\n")

        # ----- Reasoning: use task + profiles to rank candidate_list -----
        reasoning_context = {
            "task": test_task,
            "user_profile": user_profile,
            "item_profiles": item_profiles,
            "plan": plan,
            "candidate_list": test_task["candidate_list"],
        }
        reasoning_prompt = (
            "You are a recommendation system. "
            "Given the JSON context below, rank all items in candidate_list from most to least preferred "
            "for the user. Return ONLY a Python list of item_id strings, in descending preference order, "
            "and each id MUST come from candidate_list. Do not include any explanations.\n\n"
            f"CONTEXT:\n{json.dumps(reasoning_context, ensure_ascii=False)}"
        )

        reasoning_output = self.reasoning(reasoning_prompt)
        print("===== RAW REASONING OUTPUT =====")
        print(reasoning_output)

        # Parse ranked candidate list from reasoning_output
        ranked_list = []
        try:
            match = re.search(r"\[.*?\]", reasoning_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                parsed = eval(list_str)  # simple eval for quick experiment
                if isinstance(parsed, list):
                    ranked_list = parsed
        except Exception:
            ranked_list = []

        # Sanitize: keep only candidate_list ids, preserve order, deduplicate
        seen = set()
        ranked_filtered = []
        for cid in ranked_list:
            if (
                isinstance(cid, str)
                and cid in test_task["candidate_list"]
                and cid not in seen
            ):
                seen.add(cid)
                ranked_filtered.append(cid)
        # append any missing candidates at the end in original order
        for cid in test_task["candidate_list"]:
            if cid not in seen:
                ranked_filtered.append(cid)

        print("\n[Ranked list from reasoning]:")
        print(ranked_filtered)

        # ----- Pairwise Reranking on top-K candidates -----
        user_id = test_task.get("user_id")
        raw_user_info = self.interaction_tool.get_user(user_id=user_id)
        user_reviews = self.interaction_tool.get_reviews(user_id=user_id)

        clean_history = []
        if isinstance(user_reviews, list):
            for r in user_reviews[:10]:
                clean_history.append(
                    {
                        "item_id": r.get("item_id"),
                        "rating": r.get("stars", r.get("rating", "N/A")),
                        "text": r.get("text", "")[:200],
                    }
                )

        rich_user_profile = {
            "basic_info": raw_user_info,
            "history_reviews": clean_history,
            "instruction": "Please infer user taste from history_reviews.",
        }

        # Fetch item profiles for top-K items (using raw item metadata)
        top_k = 5
        candidates_to_fetch = ranked_filtered[:top_k]
        item_profiles_for_pairwise = []
        for item_id in candidates_to_fetch:
            info = self.interaction_tool.get_item(item_id=item_id)
            if info:
                info["item_id"] = item_id
                item_profiles_for_pairwise.append(info)

        pairwise_context = {
            "user_profile": rich_user_profile,
            "item_profiles": item_profiles_for_pairwise,
        }

        print(
            f"[BaselineTest] Pairwise reranking on top {len(item_profiles_for_pairwise)} candidates..."
        )
        final_ranking = self.pairwise_ranker.rerank(ranked_filtered, pairwise_context)

        print("\n[Ranked list after pairwise rerank]:")
        print(final_ranking)

        # eval against groundtruth: check rank position before and after pairwise
        pre_rank_pos = None
        if groundtruth_item in ranked_filtered:
            pre_rank_pos = ranked_filtered.index(groundtruth_item) + 1  # 1-based
        pre_is_top5 = pre_rank_pos is not None and pre_rank_pos <= 5

        rank_pos = None
        if groundtruth_item in final_ranking:
            rank_pos = final_ranking.index(groundtruth_item) + 1  # 1-based
        is_top5 = rank_pos is not None and rank_pos <= 5

        print("[Ground truth]:", groundtruth_item)
        print("[Ground truth rank BEFORE pairwise]:", pre_rank_pos)
        print("[Is in top-5 BEFORE pairwise]:", pre_is_top5)
        print("[Ground truth rank AFTER pairwise]:", rank_pos)
        print("[Is in top-5 AFTER pairwise]:", is_top5)

        # Add Memory
        trajectory = (
            f"Task:\n"
            f"    {task_description}\n\n"
            f"Plan:\n"
            f"    {plan}\n\n"
            f"UserProfile:\n"
            f"    {json.dumps(user_profile, ensure_ascii=False)}\n\n"
            f"RankedList:\n"
            f"    {final_ranking}\n\n"
            f"GroundTruth:\n"
            f"    {groundtruth_item}\n\n"
            f"RankPosBeforePairwise:\n"
            f"    {pre_rank_pos}\n\n"
            f"IsTop5BeforePairwise:\n"
            f"    {pre_is_top5}\n\n"
            f"RankPosAfterPairwise:\n"
            f"    {rank_pos}\n\n"
            f"IsTop5AfterPairwise:\n"
            f"    {is_top5}\n"
        )
        print("\n[Trajectory]:", trajectory)

        # save summary for external aggregation
        self.last_result_summary = {
            "task_name": task_name if task_name is not None else "task_0",
            "groundtruth_item": groundtruth_item,
            "pre_rank_pos": pre_rank_pos,
            "pre_is_top5": pre_is_top5,
            "post_rank_pos": rank_pos,
            "post_is_top5": is_top5,
        }

        return final_ranking


if __name__ == "__main__":
    # api_key_gemini = GEMINI_KEY
    # api_key_openai = OPENAI_API_KEY
    api_key_google = GOOGLE_API_KEY
    # llm_gemini = GeminiLLM(api_key=api_key_gemini, model="gemini-1.5-pro")
    # llm_openai = OpenAILLM(api_key=api_key_openai, model="gpt-4o")
    llm_google = GeminiLLM(api_key=api_key_google, model="gemini-2.5-flash")

    dataset = "goodreads"
    print(
        "\n===== TEST: Planning Module (Recommendation Voyager, Baseline with Pairwise) =====\n"
    )
    agent = TestRecommendationAgent(llm_google, dataset=dataset)

    # collect all tasks for this dataset
    task_dir = f"./example/track2/{dataset}/tasks"
    gt_dir = f"./example/track2/{dataset}/groundtruth"
    task_files = sorted(
        [
            f
            for f in os.listdir(task_dir)
            if f.startswith("task_") and f.endswith(".json")
        ],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not task_files:
        raise FileNotFoundError(f"No task files found in {task_dir}")

    # randomly sample a few tasks (no threading here, so keep small)
    k = 5
    seed = 42
    random.seed(seed)
    indices = random.sample(range(len(task_files)), k=min(k, len(task_files)))

    summaries = []
    for idx in indices:
        task_file = task_files[idx]
        task_index_str = task_file.split("_")[1].split(".")[0]
        task_name = f"task_{task_index_str}"

        task_path = os.path.join(task_dir, task_file)
        with open(task_path, "r", encoding="utf-8") as f:
            test_task = json.load(f)

        gt_path = os.path.join(gt_dir, f"groundtruth_{task_index_str}.json")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        groundtruth_item = gt.get("ground truth")

        print(f"\n===== RUNNING {task_name} =====")
        result = agent.workflow(
            test_task=test_task,
            groundtruth_item=groundtruth_item,
            task_name=task_name,
        )
        print("Final recommended list:", result)

        if agent.last_result_summary is not None:
            summaries.append(agent.last_result_summary)

    # print per-task ranks before/after pairwise
    if summaries:
        print("\n===== PER-TASK GROUNDTRUTH RANK (PRE / POST PAIRWISE) =====")
        for s in summaries:
            print(
                f"{s['task_name']}: "
                f"pre_rank={s['pre_rank_pos']}, pre_top5={s['pre_is_top5']}; "
                f"post_rank={s['post_rank_pos']}, post_top5={s['post_is_top5']}"
            )
