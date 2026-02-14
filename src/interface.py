from abc import ABC, abstractmethod

class SearchEngine(ABC):
    """Interface that any search backend must implement to be evaluated."""

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> list[str]:
        """
        Search for opinions relevant to the given query.

        Args:
            query: The search query string (keyword, natural language, or fact pattern).
            top_k: Maximum number of results to return.

        Returns:
            A list of opinion IDs (e.g., ["A-24-003", "89-142", "75003"]),
            ordered by relevance (most relevant first).
        """
        pass

    def name(self) -> str:
        """Human-readable name for this search engine (used in reports)."""
        return self.__class__.__name__
