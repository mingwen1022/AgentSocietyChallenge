"""Base classes for parameter retrieval."""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger("websocietysimulator")


class ParameterRetrieverBase:
    """Base class with shared utilities."""
    
    def __init__(self, llm=None, interaction_tool=None):
        self.llm = llm
        self.interaction_tool = interaction_tool
    
    def _parse_llm_response(self, response: str) -> List[str]:
        response = response.strip()
        if "```" in response:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                response = match.group(0)
        
        for pattern in [response, re.search(r'\[[^\[\]]*(?:\{[^}]*\}[^\[\]]*)*\]', response, re.DOTALL)]:
            if isinstance(pattern, re.Match):
                pattern = pattern.group(0)
            try:
                params = json.loads(pattern)
                if isinstance(params, list):
                    return [p.strip().lower().replace(" ", "_").replace("-", "_") 
                           for p in params if isinstance(p, str) and p.strip()]
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                continue
        return []
    
    def _format_reviews(self, reviews: List[Dict[str, Any]]) -> str:
        if not reviews:
            return "No reviews available"
        summaries = [f"Rating: {r.get('stars', '')}/5, Review: {r.get('text', '')[:200]}" 
                    for r in reviews if r.get("text")]
        return "\n".join(summaries[:5])

