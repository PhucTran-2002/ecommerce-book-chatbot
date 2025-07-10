import logging
from typing import Any, Dict, List

from flask import current_app
from pinecone.grpc import PineconeGRPC as Pinecone
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and similarity search with Pinecone"""

    def __init__(self):
        self.model = None
        self.index = None
        self.initialized = False

    def initialize(self):
        """Initialize Pinecone and embedding model"""
        try:
            pinecone_api_key = current_app.config.get("PINECONE_API_KEY")
            if pinecone_api_key:
                self.pc = Pinecone(api_key=pinecone_api_key)
                index_name = current_app.config["PINECONE_INDEX_NAME"]
                self.index = self.pc.Index(index_name)
                logger.info("Vector service initialized with Pinecone")
            else:
                logger.warning("PINECONE_API_KEY not found. Vector service will use fallback search.")
                self.index = None

            self.model = SentenceTransformer(current_app.config["EMBEDDING_MODEL"])

            self.initialized = True
            logger.info("Vector service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector service: {str(e)}")
            # Continue without Pinecone for now
            self.index = None
            self.initialized = True

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        if not self.initialized:
            self.initialize()

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def upsert_product_embedding(
        self, product_id: str, text: str, metadata: Dict[str, Any] = None
    ):
        """Store product embedding in Pinecone"""
        if not self.initialized:
            self.initialize()

        try:
            embedding = self.generate_embedding(text)

            vector_data = {
                "id": product_id,
                "values": embedding,
                "metadata": metadata or {},
            }

            self.index.upsert([vector_data])
            logger.info(f"Upserted embedding for product: {product_id}")

        except Exception as e:
            logger.error(f"Failed to upsert product embedding: {str(e)}")
            raise

    def search_similar_products(
        self, query_text: str, top_k: int = 10, filter_dict: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar products using vector similarity"""
        if not self.initialized:
            self.initialize()

        try:
            if not self.index:
                # Fallback to basic text search when Pinecone is not available
                logger.info("Using fallback search (Pinecone not available)")
                return self._fallback_search(query_text, top_k)

            query_embedding = self.generate_embedding(query_text)

            search_kwargs = {
                "vector": query_embedding,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False,
            }

            if filter_dict:
                search_kwargs["filter"] = filter_dict

            results = self.index.query(**search_kwargs)

            similar_products = []
            for match in results["matches"]:
                similar_products.append(
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "metadata": match.get("metadata", {}),
                    }
                )

            logger.info(
                f"Found {len(similar_products)} similar products for query: {query_text}"
            )
            return similar_products

        except Exception as e:
            logger.error(f"Failed to search similar products: {str(e)}")
            return self._fallback_search(query_text, top_k)

    def delete_product_embedding(self, product_id: str):
        """Delete product embedding from Pinecone"""
        if not self.initialized:
            self.initialize()

        try:
            self.index.delete(ids=[product_id])
            logger.info(f"Deleted embedding for product: {product_id}")

        except Exception as e:
            logger.error(f"Failed to delete product embedding: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.initialized:
            self.initialize()

        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {}

    def batch_upsert_products(
        self, products: List[Dict[str, Any]], batch_size: int = 100
    ):
        """Batch upsert multiple product embeddings"""
        if not self.initialized:
            self.initialize()

        try:
            vectors = []

            for product in products:
                embedding = self.generate_embedding(product["text"])
                vectors.append(
                    {
                        "id": product["id"],
                        "values": embedding,
                        "metadata": product.get("metadata", {}),
                    }
                )

                if len(vectors) >= batch_size:
                    self.index.upsert(vectors)
                    vectors = []

            if vectors:
                self.index.upsert(vectors)

            logger.info(f"Batch upserted {len(products)} product embeddings")

        except Exception as e:
            logger.error(f"Failed to batch upsert products: {str(e)}")
            raise

    def _fallback_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback search when Pinecone is not available"""
        try:
            from models.product import Product
            
            # Simple text-based search
            products = Product.query.filter(
                Product.is_active == True
            ).limit(top_k).all()
            
            # Simple scoring based on text matching
            scored_products = []
            query_lower = query_text.lower()
            
            for product in products:
                score = 0
                if query_lower in product.name.lower():
                    score += 2
                if query_lower in product.description.lower():
                    score += 1
                if query_lower in product.category.lower():
                    score += 1
                if query_lower in product.brand.lower():
                    score += 1
                
                if score > 0:
                    scored_products.append({
                        "id": product.id,
                        "score": score,
                        "metadata": {
                            "category": product.category,
                            "subcategory": product.subcategory,
                            "brand": product.brand,
                            "price": product.price,
                            "rating": product.rating,
                            "in_stock": product.is_in_stock(),
                        }
                    })
            
            # Sort by score and return top_k
            scored_products.sort(key=lambda x: x["score"], reverse=True)
            return scored_products[:top_k]
            
        except Exception as e:
            logger.error(f"Error in fallback search: {str(e)}")
            return []
