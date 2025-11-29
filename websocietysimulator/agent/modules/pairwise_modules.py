import json


class PairwiseRanker:
    def __init__(self, llm):
        self.llm = llm
        self.K = 5  # Only rerank the top 5 items
        # With "King of the Hill", this costs K-1 LLM calls per task.

    def rerank(self, initial_ranking, context):
        """
        Input:
            initial_ranking: List of item IDs (e.g., length 20)
            context: Dict containing 'user_profile' and 'item_profiles'
        Output:
            final_ranking: List of item IDs (winner moved to #1, others keep order)
        """
        item_map = {item["item_id"]: item for item in context.get("item_profiles", [])}
        user_profile = context.get("user_profile", {})

        candidates = initial_ranking[: self.K]
        rest = initial_ranking[self.K :]

        if len(candidates) < 2:
            return initial_ranking

        current_winner_id = candidates[0]
        print(
            f"\n[Pairwise] Starting 'King of the Hill' scan on Top {self.K} candidates..."
        )

        for challenger_id in candidates[1:]:
            profile_winner = item_map.get(current_winner_id)
            profile_challenger = item_map.get(challenger_id)

            if not profile_winner or not profile_challenger:
                continue

            winner_result = self.compare_pair(
                user_profile, profile_winner, profile_challenger
            )

            winner_label = (
                "Challenger" if winner_result == challenger_id else "Current King"
            )
            print(
                f"[DEBUG] {current_winner_id} (King) vs {challenger_id} (Challenger) -> Winner: {winner_label}"
            )

            if winner_result == challenger_id:
                current_winner_id = challenger_id

        print(f"[Pairwise] Final Winner identified: {current_winner_id}")

        others = [x for x in candidates if x != current_winner_id]
        final_top_k = [current_winner_id] + others
        final_ranking = final_top_k + rest

        final_ranking = [str(item) for item in final_ranking]
        if len(final_ranking) != len(initial_ranking):
            print(
                f"[ERROR] List length changed! Original: {len(initial_ranking)}, New: {len(final_ranking)}"
            )
            return initial_ranking

        return final_ranking

    def compare_pair(self, user_profile, item_a, item_b):
        """
        Compare two items (A vs B) for a given user, return winning item_id.
        """
        try:
            user_str = json.dumps(user_profile, ensure_ascii=False)
            item_a_str = json.dumps(item_a, ensure_ascii=False)
            item_b_str = json.dumps(item_b, ensure_ascii=False)

            prompt = f"""
Role: You are a strict personal librarian and recommendation judge.
Your goal is to determine if a new book candidate is **significantly better** than the current best option for this specific user.

[User Profile]
{user_str}

[Candidate B (The Challenger)]
{item_b_str}

[Candidate A (The Current Champion)]
{item_a_str}

**Task:**
Compare Candidate A and Candidate B strictly based on the user's taste history.
You must decide if Candidate B provides a specific value that Candidate A misses.

**Strict Evaluation Rules:**
1. **Threshold:** Only pick Candidate B if it is a **SIGNIFICANTLY better fit** for the user's specific sub-genres or mood than Candidate A.
2. **Tie-Breaker:** If both books are equally good fits, or if the difference is subjective, **you MUST stick with Candidate A**.
3. **Niche over Popularity:** Do not pick B just because it is popular. It must match the user's unique history.

**Output:**
Reasoning: [Explain why B is/is not significantly better than A]
Winner: [A or B]
"""
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(messages=messages)

            print(f"[LLM Thinking]: {response}")
            response = response.strip().upper()

            id_a, id_b = item_a["item_id"], item_b["item_id"]

            if "WINNER: B" in response or "WINNER: [B]" in response:
                return id_b
            if "WINNER: A" in response or "WINNER: [A]" in response:
                return id_a

            clean_response = response.replace(".", "").strip()
            if clean_response.endswith("WINNER: B") or clean_response.endswith(" B"):
                return id_b
            if clean_response.endswith("WINNER: A") or clean_response.endswith(" A"):
                return id_a

            last_line = response.split("\n")[-1]
            if "B" in last_line and "A" not in last_line:
                return id_b
            if "A" in last_line and "B" not in last_line:
                return id_a

            return id_a
        except Exception:
            return None

