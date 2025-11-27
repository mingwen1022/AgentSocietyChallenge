"""User and Item parameter retrievers for dynamic profile parameter extraction."""

import logging
from typing import Any, Dict, List, Optional

from websocietysimulator.agent.modules.parameter_retriever_base import ParameterRetrieverBase

logger = logging.getLogger("websocietysimulator")


class UserParameterRetriever(ParameterRetrieverBase):
    """Extracts dynamic parameters for user profiles by analyzing review history."""
    
    PARAMETERS = [
        "genre_preference", "reading_style", "theme_preference", 
        "author_preference", "review_sentiment", "character_preference",
        "plot_complexity", "writing_style_preference"
    ]
    DEFAULT_PARAMETERS = PARAMETERS[:2]
    
    def retrieve_parameters(self, user_id: str, steps: List[Dict[str, Any]]) -> List[str]:
        """Extract dynamic parameters from user's review history."""
        if not self.llm or not self.interaction_tool or not user_id:
            return self.DEFAULT_PARAMETERS
        
        try:
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            if not user_reviews:
                return self.DEFAULT_PARAMETERS
            
            review_summaries = [
                f"Rating: {r.get('stars', '')}/5, Review: {r.get('text', '')[:200]}"
                for r in user_reviews[:20] if r.get("text")
            ]
            review_context = "\n".join(review_summaries[:10])
            step_context = "\n".join(f"- {s.get('description', '')}" for s in steps[:3])
            
            prompt = f"""Analyze user review history to determine profile parameters.

Reviews ({len(user_reviews)} total):
{review_context}

Planner Steps:
{step_context}

Determine 2-4 parameters (snake_case) capturing preferences.
Look for: genre, author, themes, characters, plot, writing style.

Examples: {', '.join(self.PARAMETERS)}
Return JSON: ["genre_preference", "character_preference", "plot_complexity"]"""
            
            params = self._parse_llm_response(self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.3))
            if params:
                return params
            return self.DEFAULT_PARAMETERS
        except Exception as e:
            logger.error(f"  [UserParameterRetriever] Error: {e}")
            return self.DEFAULT_PARAMETERS


class ItemParameterRetriever(ParameterRetrieverBase):
    """Extracts dynamic parameters for item profiles by analyzing metadata, reviews, and candidates."""
    
    PARAMETERS = [
        "genre", "reading_level", "theme", "author_style", "rating",
        "target_audience", "writing_style", "plot_type"
    ]
    DEFAULT_PARAMETERS = PARAMETERS[:2]
    
    def retrieve_parameters(
        self, 
        item_id: str, 
        steps: List[Dict[str, Any]], 
        candidate_list: Optional[List[str]] = None
    ) -> List[str]:
        """Extract dynamic parameters from item metadata, reviews, and candidate comparison."""
        if not self.llm or not self.interaction_tool or not item_id:
            return self.DEFAULT_PARAMETERS
        
        try:
            item_record = self.interaction_tool.get_item(item_id)
            if not item_record:
                return self.DEFAULT_PARAMETERS
            
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)
            item_summary = self._format_item_metadata(item_record)
            review_context = self._format_reviews(item_reviews[:10])
            candidate_analysis = self._format_candidates(candidate_list) if candidate_list and len(candidate_list) > 1 else ""
            step_context = "\n".join(f"- {s.get('description', '')}" for s in steps[:3])
            
            prompt = f"""Analyze item metadata and reviews to determine profile parameters.

Item:
{item_summary}

Reviews:
{review_context}
{candidate_analysis}
Planner Steps:
{step_context}

Determine 2-4 parameters (snake_case) capturing characteristics and distinguishing from candidates.
Look for: genre, themes, writing style, target audience, reviewer mentions.

Examples: {', '.join(self.PARAMETERS)}
Return JSON: ["genre", "theme", "writing_style"]"""
            
            params = self._parse_llm_response(self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.3))
            if params:
                return params
            return self.DEFAULT_PARAMETERS
        except Exception as e:
            logger.error(f"  [ItemParameterRetriever] Error: {e}")
            return self.DEFAULT_PARAMETERS
    
    def _format_item_metadata(self, item: Dict[str, Any]) -> str:
        parts = [f"Title: {item.get('title', 'N/A')}"]
        if item.get('authors'):
            parts.append(f"Authors: {self._format_field(item.get('authors'))}")
        if item.get('description'):
            parts.append(f"Description: {item.get('description', '')[:300]}")
        if item.get('average_rating'):
            parts.append(f"Rating: {item.get('average_rating')}")
        if item.get('popular_shelves'):
            parts.append(f"Shelves: {self._format_shelves(item.get('popular_shelves'))}")
        return "\n".join(parts)
    
    def _format_field(self, field: Any) -> str:
        if isinstance(field, list):
            return ', '.join(str(f) for f in field)
        elif isinstance(field, dict):
            return ', '.join(str(v) for v in field.values())
        return str(field)
    
    def _format_shelves(self, shelves: Any) -> str:
        if isinstance(shelves, list):
            return ', '.join(str(s) if not isinstance(s, dict) else str(s.get('name', s)) for s in shelves)
        return str(shelves)
    
    def _format_candidates(self, candidate_list: List[str]) -> str:
        candidate_items = []
        for cand_id in candidate_list[:5]:
            cand_item = self.interaction_tool.get_item(cand_id)
            if cand_item:
                shelves = cand_item.get('popular_shelves', [])
                if isinstance(shelves, list):
                    genre_list = [str(s) if not isinstance(s, dict) else str(s.get('name', s)) for s in shelves[:3]]
                else:
                    genre_list = [str(shelves)] if shelves else []
                candidate_items.append({
                    'title': cand_item.get('title', 'N/A'),
                    'authors': cand_item.get('authors', 'N/A'),
                    'genre': genre_list
                })
        
        if not candidate_items:
            return ""
        
        candidate_summary = "\n".join(
            f"- {item['title']} by {self._format_field(item['authors'])} (genres: {', '.join(item['genre'])})"
            for item in candidate_items
        )
        return f"\nCandidates (for comparison):\n{candidate_summary}\n\nIdentify parameters that distinguish the target item from these candidates."

