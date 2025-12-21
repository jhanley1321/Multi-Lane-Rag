

    def search(self, query: str, lane_name: Optional[str] = None, k: int = 3) -> List:
        """
        Search for relevant documents in the vector database.
        
        Args:
            query: The search query
            lane_name: Optional lane to search in (searches all if None)
            k: Number of documents to return
        
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded. Please initialize the database first.")
            return []
        
        results = self.vectorstore.similarity_search(query=query, k=k)
        
        if lane_name:
            results = [doc for doc in results if doc.metadata.get('lane') == lane_name]
        
        return results
