# planning_recommendation_voyager.py
"""
A custom Planning module for recommendation tasks,
built on top of PlanningVoyager but with additional
field descriptions and task goals.
"""

import re
import ast
from websocietysimulator.agent.modules.planning_modules import PlanningBase

# yelp
yelp_guides = """
You are solving a RECOMMENDATION planning task.

Input task fields (Yelp):
-type: is's fixed as "recommendation"
- user_id: appears in user and review
- candidate_list: a list of item_id; each appears in item and review
- candidate_category: high-level category of candidate items
- loc: (latitude, longitude)

Relevant datasets (Yelp):

user dataset — one row per user:
- user_id: unique user identifier; links to review.user_id
- name: user nickname
- review_count: total number of reviews this user has written
- yelping_since: when the user joined Yelp
- useful / funny / cool: votes received for their reviews
- elite: elite years string or None
- friends: list of friend user_ids or None
- fans: number of fans
- average_stars: user's average rating across reviews
- compliment_hot / more / profile / cute / list / note / plain / cool /
  funny / writer / photos: compliment counters indicating engagement level
- source: fixed as 'yelp'

item dataset — one row per business:
- item_id: unique POI identifier; links to review.item_id
- name: business name
- address: street address
- city / state / postal_code: business location metadata
- latitude / longitude: coordinates for distance matching
- stars: average rating for business
- review_count: number of reviews for business
- is_open: 1 = open, 0 = closed
- attributes: key-value dict of business properties
    (e.g., Alcohol, HasTV, Ambience, NoiseLevel, WiFi, etc.)
- categories: comma-separated high-level tags (e.g., "Restaurants, Barbeque")
- hours: daily opening times dictionary
- source: fixed as 'yelp'
- type: object type, usually "business"

review dataset — one row per review:
- review_id: unique review identifier
- user_id: reviewer (matches user.user_id)
- item_id: reviewed business (matches item.item_id)
- stars: star rating (1-5)
- useful / funny / cool: votes on the review
- text: free-text content of user's review
- date: timestamp of review
- source: fixed as 'yelp'
- type: object type, usually "business review"

Your goal:
1. Generate subgoals that retrieve necessary information from 
   user, item, and review.
2. Ensure subgoals enable a reasoning module to later rank 
   the candidate_list from most relevant to least relevant for the user.
3. Keep subgoals high-level, concise, and logically ordered.
4. For each subgoal, output both "description" and "reasoning instruction".
"""

good_reads_guides = """
You are solving a RECOMMENDATION planning task.

Input task fields (Goodreads):
- type: it's fixed as "recommendation"
- user_id: appears in user and review datasets
- candidate_list: list of item_id (books) that appear in item and review datasets
- candidate_category: high-level genre/category of candidate books
- loc: (latitude, longitude) — usually unused for Goodreads but remains for interface consistency

Relevant datasets (Goodreads):

user dataset — one row per user:
- user_id: unique user identifier; links to review.user_id
- source: fixed as 'goodreads'
(Users often contain minimal metadata; profile fields may not be available.)

item dataset — one row per book:
- item_id: unique book identifier; links to review.item_id
- title: book title
- title_without_series: cleaned title without series annotation
- average_rating: aggregated community rating
- ratings_count: total number of ratings received
- text_reviews_count: total number of textual reviews
- description: book summary or synopsis
- authors: list of authors with author_id and role
- publisher: publishing company
- num_pages: number of pages
- publication_year / month / day: publication date metadata
- isbn / isbn13: book identifiers
- asin / kindle_asin: Amazon identifiers if available
- format: book format (paperback, ebook, etc.)
- is_ebook: boolean flag
- popular_shelves: list of user-defined shelves with counts 
    (e.g., to-read, currently-reading, favorites)
- similar_books: list of related/recommended book ids
- country_code: country metadata
- language_code: language of the book
- link / url: Goodreads detail page links
- image_url: cover image URL
- work_id: Goodreads work identifier
- source: fixed as 'goodreads'
- type: object type, usually "book"

review dataset — one row per review:
- review_id: unique review identifier
- user_id: reviewer id; links to user.user_id
- item_id: reviewed book id; links to item.item_id
- stars: star rating (float)
- text: textual review content (very important for preference extraction)
- date_added / date_updated: timestamps of review creation & update
- read_at: when the user marked the book as read (may be empty)
- started_at: reading start time (may be empty)
- n_votes: number of likes/upvotes the review received
- n_comments: number of comments on the review
- source: fixed as 'goodreads'
- type: usually 'book'

Your goal:
1. Generate subgoals that retrieve necessary information from 
   user, item, and review datasets.
2. Ensure subgoals enable a reasoning module to later rank 
   the candidate_list from most relevant to least relevant for the user.
3. Keep subgoals high-level, concise, and logically ordered.
4. For each subgoal, output both "description" and "reasoning instruction".
"""

FIELD_GUIDE = good_reads_guides

OUTPUT_STYLE_GUIDE = """
You must output a list of subgoals in the following style:
sub-task 1: {"description": "...", "reasoning instruction": "..."}
sub-task 2: {"description": "...", "reasoning instruction": "..."}
"""


def build_few_shot_block(few_shot):
    if few_shot in [None, "", []]:
        return ""
    return f"""
Here are successful examples to guide your planning:
{few_shot}

"""


class PlanningVoyagerCustom(PlanningBase):

    def create_prompt(self, task_type, task_description, feedback, few_shot):
        """
        Enhanced Voyager-style planner:
        - keeps the subgoal-generation style
        - adds lightweight field descriptions
        - adds recommendation task goals
        """
        # relevant datasets need to add instruction for each features

        # build few-shot block if provided
        FEW_SHOT_BLOCK = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task.

{FIELD_GUIDE}

{FEW_SHOT_BLOCK}

Task: {task_description}

{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task.

{FIELD_GUIDE}

{FEW_SHOT_BLOCK}

end
--------------------
reflexion:{feedback}

Task: {task_description}

{OUTPUT_STYLE_GUIDE}
"""

        return prompt


class PlanningIOCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}

Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningDEPSCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helper AI agent in reasoning. You need to generate the sequences of sub-goals (actions) for a {task_type} task in multi-hop questions. You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningTDCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format must follow the example below, specifying the order and dependencies.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks with explicit temporal dependencies.
Consider the order of actions and their dependencies to ensure logical sequencing.
Your output format should follow the example below, specifying the order and dependencies.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningVoyagerCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a helpful assistant that generates subgoals to complete any {task_type} task specified by me.
I'll give you a final task, you need to decompose the task into a list of subgoals.
You must follow the following criteria:
1) Return a list of subgoals that can be completed in order to complete the specified task.
2) Give the reasoning instructions for each subgoal and the instructions for calling the tool. 
You also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
reflexion:{feedback}
task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningOPENAGICustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.  
Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who is an expert at coming up with a todo list for a given {task_type} objective.
For each task, you also need to give the reasoning instructions for each subtask and the instructions for calling the tool.
Ensure the list is as short as possible, and tasks in it are relevant, effective and described in a single sentence.
Develop a concise to-do list to achieve the objective.
Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt


class PlanningHUGGINGGPTCustom(PlanningBase):
    def create_prompt(self, task_type, task_description, feedback, few_shot):
        few_shot_block = build_few_shot_block(few_shot)

        if feedback == "":
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
Task: {task_description}
{OUTPUT_STYLE_GUIDE}
"""
        else:
            prompt = f"""
You are a planner who divides a {task_type} task into several subtasks. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. you also need to give the reasoning instructions for each subtask and the instructions for calling the tool. Your output format should follow the example below.

{FIELD_GUIDE}
{few_shot_block}
end
--------------------
Reflexion:{feedback}
Task:{task_description}
{OUTPUT_STYLE_GUIDE}
"""
        return prompt
