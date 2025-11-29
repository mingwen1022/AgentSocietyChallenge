# rec_agent_experiment/rec_agent_experiment_simulator_test.py
import random
import sys
import os
import json
import logging
import re
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.reasoning_modules import *
from websocietysimulator.agent.modules.info_orchestrator_module import InfoOrchestrator
from websocietysimulator.agent.modules.schemafitter_module import SchemaFitterIO
from websocietysimulator.agent.modules.pairwise_modules import PairwiseRanker
from gemini import GeminiLLM
from Ai_AGENT_SH_branch.websocietysimulator.agent.modules.planning_modules import (
    PlanningIO as PlanningIO_SH,
    PlanningVoyager as PlanningVoyager_SH,
    PlanningDEPS as PlanningDEPS_SH,
)

from websocietysimulator.agent.modules.planning_module_custom import *
from websocietysimulator.agent.modules.memory_modules_custom import *

log_file = f"logs/sim_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Pairwise reranking configuration (set USE_PAIRWISE=True to enable)
USE_PAIRWISE = True
PAIRWISE_TOP_K = 5


class MyRecommendationAgent(RecommendationAgent):
    """
    Recommendation agent for track2_test using:
    - PlanningIOCustom as planner
    - InfoOrchestrator + SchemaFitterIO for user/item profiles
    - MemoryDILU as long-term memory (trajectory storage)
    - Reasoning to rank candidate_list
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.planning = PlanningVoyagerCustom(llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)
        self.reasoning = ReasoningIO(
            profile_type_prompt="You are an intelligent recommendation system.",
            memory=None,
            llm=self.llm,
        )
        # InfoOrchestrator; schema_fitter and interaction_tool will be set later
        self.info_orchestrator = InfoOrchestrator(
            memory=None,
            llm=self.llm,
            schema_fitter=None,
            interaction_tool=None,
            use_fixed_item_params=True,
            max_candidates_to_profile=None,
        )
        self._schema_fitter_llm = self.llm
        # Optional pairwise reranker on top-K items
        self.pairwise_ranker = PairwiseRanker(self.llm) if USE_PAIRWISE else None
        logging.info("=== Experiment config ===")
        logging.info("Planning module: %s", type(self.planning).__name__)
        logging.info("Reasoning module: %s", type(self.reasoning).__name__)
        logging.info("Memory module: %s", type(self.memory).__name__)
        logging.info("Use pairwise rerank: %s", USE_PAIRWISE)

    def set_interaction_tool(self, interaction_tool):
        """
        Override to also wire interaction_tool into InfoOrchestrator.
        """
        super().set_interaction_tool(interaction_tool)
        if self.info_orchestrator and interaction_tool:
            if self.info_orchestrator.schema_fitter is None:
                schema_fitter = SchemaFitterIO(
                    self._schema_fitter_llm, interaction_tool
                )
                self.info_orchestrator.schema_fitter = schema_fitter
            self.info_orchestrator.interaction_tool = interaction_tool

    def workflow(self):
        """
        Main workflow for a single recommendation task.
        Simulator will call this once per task.

        Returns:
            list[str]: Ranked list of candidate item IDs.
        """
        # Current task is already converted to dict by RecommendationAgent.insert_task
        task = self.task
        logging.info(
            "---- Task start ---- user_id=%s  #cands=%d",
            task.get("user_id"),
            len(task.get("candidate_list", [])),
        )
        # 1) Build task description
        task_description = json.dumps(task, indent=2)

        # 3) Retrieve few-shot examples from memory (if any)
        task_query = json.dumps(task, ensure_ascii=False)
        few_shot = self.memory(task_query) or ""

        # 4) Call planner to generate sub-tasks
        plan = self.planning(
            task_type="Recommendation Task",
            task_description=task_description,
            feedback="",
            few_shot=few_shot,
        )

        # 5) Build user/item profiles via InfoOrchestrator
        profiles = self.info_orchestrator(
            planner_steps=plan,
            user_id=task.get("user_id"),
            candidate_list=task.get("candidate_list"),
        )
        user_profile = profiles.get("user_profile")
        item_profiles = profiles.get("item_profiles", [])

        # 6) Reasoning: rank candidate_list
        reasoning_context = {
            "task": task,
            "plan": plan,
            "user_profile": user_profile,
            "item_profiles": item_profiles,
            "candidate_list": task["candidate_list"],
        }
        reasoning_prompt = (
            "You are a recommendation system. "
            "Given the JSON context below, rank all items in candidate_list from most to least preferred "
            "for the user. Return ONLY a Python list of item_id strings, in descending preference order, "
            "and each id MUST come from candidate_list. Do not include any explanations. "
            "Your entire output must be very short (no more than 1 line, strictly no analysis).\n\n"
            f"CONTEXT:\n{json.dumps(reasoning_context, ensure_ascii=False)}"
        )

        reasoning_output = self.reasoning(reasoning_prompt)
        logging.info("RAW REASONING OUTPUT:\n%s", reasoning_output)

        # Parse ranked candidate list from reasoning output
        ranked_list = []
        try:
            match = re.search(r"\[.*?\]", reasoning_output, re.DOTALL)
            if match:
                list_str = match.group(0)
                parsed = eval(list_str)
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
                and cid in task["candidate_list"]
                and cid not in seen
            ):
                seen.add(cid)
                ranked_filtered.append(cid)
        # append any missing candidates at the end in original order
        for cid in task["candidate_list"]:
            if cid not in seen:
                ranked_filtered.append(cid)
        logging.info("RANKED LIST BEFORE PAIRWISE (len=%d)", len(ranked_filtered))
        logging.info("Ranked list (pre-pairwise): %s", ranked_filtered)

        # Optional Pairwise Reranking on top-K candidates
        if self.pairwise_ranker and self.interaction_tool:
            user_id = task.get("user_id")
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

            # Use existing item_profiles from InfoOrchestrator for top-K items
            top_ids = ranked_filtered[:PAIRWISE_TOP_K]
            item_profiles_for_pairwise = []
            for item_id in top_ids:
                info = self.interaction_tool.get_item(item_id=item_id)
                if info:
                    info["item_id"] = item_id
                    item_profiles_for_pairwise.append(info)

            pairwise_context = {
                "user_profile": rich_user_profile,
                "item_profiles": item_profiles_for_pairwise,
            }

            logging.info(
                "[Pairwise] Reranking top %d candidates (have %d profiles)",
                PAIRWISE_TOP_K,
                len(item_profiles_for_pairwise),
            )
            ranked_filtered = self.pairwise_ranker.rerank(
                ranked_filtered, pairwise_context
            )
            logging.info("RANKED LIST AFTER PAIRWISE (len=%d)", len(ranked_filtered))
            logging.info("Ranked list (post-pairwise): %s", ranked_filtered)

        logging.info("FINAL RANKED LIST (len=%d)", len(ranked_filtered))
        logging.info("Ranked list : %s", ranked_filtered)
        logging.info("---- Task end ----")
        return ranked_filtered


if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY in environment variables")

    # Use a small test split: example/track2_test/goodreads
    task_set = "goodreads"
    task_dir = f"./example/track2/{task_set}/tasks"
    groundtruth_dir = f"./example/track2/{task_set}/groundtruth"

    # Initialize Simulator
    # data_dir should point to where item.json / user.json / review.json are stored
    simulator = Simulator(data_dir="./data", device="auto", cache=False)

    # Load scenarios from track2_test
    simulator.set_task_and_groundtruth(
        task_dir=task_dir,
        groundtruth_dir=groundtruth_dir,
    )

    k = 50  # 想要的任务数

    # first k task
    # simulator.tasks = simulator.tasks[:k]
    # simulator.groundtruth_data = simulator.groundtruth_data[:k]

    # random k task
    seed = 82
    random.seed(seed)  # For reproducibility
    all_indices = list(range(len(simulator.tasks)))
    selected = random.sample(all_indices, k=k)
    simulator.tasks = [simulator.tasks[i] for i in selected]
    simulator.groundtruth_data = [simulator.groundtruth_data[i] for i in selected]

    # Set custom agent
    simulator.set_agent(MyRecommendationAgent)

    # Set LLM client (using Gemini here; you can swap to another LLM if needed)
    llm_google = GeminiLLM(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    simulator.set_llm(llm_google)

    # Run simulation on all tasks in track2_test
    start = time.time()
    agent_outputs = simulator.run_simulation(
        number_of_tasks=None,  # None means "run all loaded tasks"
        enable_threading=True,  # single-thread for easier debugging
        max_workers=10,
    )

    # Evaluate with groundtruth from track2_test
    evaluation_results = simulator.evaluate()
    end = time.time()

    # record run info
    total_time = end - start
    num_tasks_run = len(simulator.tasks)

    planning_name = type(MyRecommendationAgent(llm_google).planning).__name__.lower()
    reasoning_name = type(MyRecommendationAgent(llm_google).reasoning).__name__.lower()

    evaluation_results["run_info"] = {
        "planning_module": planning_name,
        "reasoning_module": reasoning_name,
        "random_seed": seed,
        "num_tasks": num_tasks_run,
        "total_time_seconds": total_time,
        "avg_time_per_task_seconds": (
            total_time / num_tasks_run if num_tasks_run > 0 else None
        ),
        "max_workers": 10,
        "max_reviews_per_profile": 70,
        "limit_on_schema_fields(prompt_instruction)": "no more than 50 English words",
    }

    # file name：rs{seed}_{planning}_{reasoning}_{k}.json
    filename = f"rs{seed}_{planning_name}_{reasoning_name}_{k}.json"
    out_path = os.path.join("./evaluation_result", filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)

    print(f"The evaluation_results is: {evaluation_results}")
    print(f"Saved evaluation results to: {out_path}")
    print(f"Simulation + evaluation for {k} tasks took {total_time:.1f} seconds")
