"""
Reranking Strategy

Implements result reranking using CrossEncoder models to improve search result ordering.
The reranking process re-scores search results based on query-document relevance using
a trained neural model, typically improving precision over initial retrieval scores.

Uses the cross-encoder/ms-marco-MiniLM-L-6-v2 model for reranking by default.
"""

import os
from typing import List, Dict, Any, Optional

# Import CrossEncoder for reranking if available
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSSENCODER_AVAILABLE = False

from ...config.logfire_config import safe_span, get_logger

logger = get_logger(__name__)

# Default reranking model
DEFAULT_RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingStrategy:
    """Strategy class implementing result reranking using CrossEncoder models"""

    def __init__(self, model_name: str = DEFAULT_RERANKING_MODEL, model_instance: Optional[CrossEncoder] = None):
        """
        Initialize reranking strategy.
        
        Args:
            model_name: Name/path of the CrossEncoder model to use
            model_instance: Pre-loaded CrossEncoder instance (optional)
        """
        self.model_name = model_name
        self.model = model_instance or self._load_model()

    def _load_model(self) -> Optional[CrossEncoder]:
        """Load the CrossEncoder model for reranking."""
        if not CROSSENCODER_AVAILABLE:
            logger.warning("sentence-transformers not available - reranking disabled")
            return None
            
        try:
            logger.info(f"Loading reranking model: {self.model_name}")
            return CrossEncoder(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load reranking model {self.model_name}: {e}")
            return None

    def is_available(self) -> bool:
        """Check if reranking is available (model loaded successfully)."""
        return self.model is not None

    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str = "content",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using the CrossEncoder model.
        
        Args:
            query: The search query used to retrieve results
            results: List of search results to rerank
            content_key: The key in each result dict containing text content for reranking
            top_k: Optional limit on number of results to return after reranking
            
        Returns:
            Reranked list of results ordered by rerank_score (highest first)
        """
        if not self.model or not results:
            logger.debug("Reranking skipped - no model or no results")
            return results

        with safe_span("rerank_results", 
                      result_count=len(results),
                      model_name=self.model_name) as span:
            try:
                # Extract text content from results for reranking
                texts = []
                valid_indices = []
                
                for i, result in enumerate(results):
                    content = result.get(content_key, "")
                    if content and isinstance(content, str):
                        texts.append(content)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"Result {i} has no valid content for reranking")

                if not texts:
                    logger.warning("No valid texts found for reranking")
                    return results

                # Create query-document pairs for the CrossEncoder
                query_doc_pairs = [[query, text] for text in texts]

                # Get reranking scores from the model
                with safe_span("crossencoder_predict"):
                    scores = self.model.predict(query_doc_pairs)

                # Add rerank scores to valid results
                for i, valid_idx in enumerate(valid_indices):
                    results[valid_idx]["rerank_score"] = float(scores[i])

                # Sort results by rerank score (descending - highest relevance first)
                reranked_results = sorted(
                    results,
                    key=lambda x: x.get("rerank_score", -1.0),
                    reverse=True
                )

                # Apply top_k limit if specified
                if top_k is not None and top_k > 0:
                    reranked_results = reranked_results[:top_k]

                span.set_attribute("reranked_count", len(reranked_results))
                span.set_attribute("score_range", f"{min(scores):.3f}-{max(scores):.3f}")

                logger.debug(f"Reranked {len(texts)} results, score range: {min(scores):.3f}-{max(scores):.3f}")
                
                return reranked_results

            except Exception as e:
                logger.error(f"Error during reranking: {e}")
                span.set_attribute("error", str(e))
                return results

    def rerank_results_sync(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str = "content",
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of rerank_results for use in non-async contexts.
        
        Args:
            query: The search query used to retrieve results
            results: List of search results to rerank  
            content_key: The key in each result dict containing text content
            top_k: Optional limit on number of results to return after reranking
            
        Returns:
            Reranked list of results ordered by rerank_score (highest first)
        """
        if not self.model or not results:
            return results

        try:
            # Extract texts and create pairs
            texts = [result.get(content_key, "") for result in results if result.get(content_key)]
            if not texts:
                return results

            query_doc_pairs = [[query, text] for text in texts]
            
            # Get scores and add to results
            scores = self.model.predict(query_doc_pairs)
            for i, result in enumerate(results[:len(scores)]):
                result["rerank_score"] = float(scores[i])

            # Sort and limit
            reranked = sorted(results, key=lambda x: x.get("rerank_score", -1.0), reverse=True)
            return reranked[:top_k] if top_k else reranked

        except Exception as e:
            logger.error(f"Error during sync reranking: {e}")
            return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded reranking model."""
        return {
            "model_name": self.model_name,
            "available": self.is_available(),
            "crossencoder_available": CROSSENCODER_AVAILABLE,
            "model_loaded": self.model is not None
        }


class RerankingConfig:
    """Configuration helper for reranking settings"""
    
    @staticmethod
    def from_credential_service(credential_service) -> Dict[str, Any]:
        """Load reranking configuration from credential service."""
        try:
            use_reranking = credential_service.get_bool_setting("USE_RERANKING", False)
            model_name = credential_service.get_setting("RERANKING_MODEL", DEFAULT_RERANKING_MODEL)
            top_k = int(credential_service.get_setting("RERANKING_TOP_K", "0"))
            
            return {
                "enabled": use_reranking,
                "model_name": model_name,
                "top_k": top_k if top_k > 0 else None
            }
        except Exception as e:
            logger.error(f"Error loading reranking config: {e}")
            return {
                "enabled": False,
                "model_name": DEFAULT_RERANKING_MODEL,
                "top_k": None
            }
    
    @staticmethod
    def from_env() -> Dict[str, Any]:
        """Load reranking configuration from environment variables."""
        return {
            "enabled": os.getenv("USE_RERANKING", "false").lower() in ("true", "1", "yes", "on"),
            "model_name": os.getenv("RERANKING_MODEL", DEFAULT_RERANKING_MODEL),
            "top_k": int(os.getenv("RERANKING_TOP_K", "0")) or None
        }


# Utility functions for standalone usage
def create_reranking_strategy(
    credential_service=None,
    model_name: Optional[str] = None
) -> Optional[RerankingStrategy]:
    """
    Create a reranking strategy instance based on configuration.
    
    Args:
        credential_service: Optional credential service for settings
        model_name: Optional model name override
        
    Returns:
        RerankingStrategy instance if enabled and available, None otherwise
    """
    try:
        if credential_service:
            config = RerankingConfig.from_credential_service(credential_service)
        else:
            config = RerankingConfig.from_env()
            
        if not config["enabled"]:
            logger.debug("Reranking disabled by configuration")
            return None
            
        effective_model_name = model_name or config["model_name"]
        strategy = RerankingStrategy(model_name=effective_model_name)
        
        if not strategy.is_available():
            logger.warning("Reranking strategy created but model not available")
            return None
            
        logger.info(f"Reranking strategy created with model: {effective_model_name}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create reranking strategy: {e}")
        return None


async def rerank_search_results(
    query: str,
    results: List[Dict[str, Any]],
    strategy: Optional[RerankingStrategy] = None,
    content_key: str = "content",
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Standalone function to rerank search results.
    
    Args:
        query: Search query
        results: Results to rerank
        strategy: Optional reranking strategy instance
        content_key: Key containing text content in results
        top_k: Optional limit on results to return
        
    Returns:
        Reranked results
    """
    if not strategy:
        strategy = create_reranking_strategy()
    
    if not strategy:
        return results
        
    return await strategy.rerank_results(query, results, content_key, top_k)