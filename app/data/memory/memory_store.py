import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict
from datetime import datetime
import uuid
import os
from app.utils.models import Memory

class MemoryStore:
    def __init__(self):
        # Create data directory if it doesn't exist
        data_dir = "./data/memory"
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=data_dir
            ))
            self.collection = self.client.get_or_create_collection("doctor_memories")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise
        
    def add_memory(self, content: str, memory_type: str, metadata: Optional[Dict] = None) -> Memory:
        memory_id = str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            metadatas=[{"type": memory_type, **metadata} if metadata else {"type": memory_type}],
            ids=[memory_id]
        )
        return Memory(
            id=memory_id,
            content=content,
            type=memory_type,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
    def search_memories(self, query: str, n_results: int = 5) -> List[Memory]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        memories = []
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            memories.append(Memory(
                id=results["ids"][0][i],
                content=doc,
                type=metadata["type"],
                timestamp=datetime.now(),  # Using current time as actual timestamp not stored
                metadata={k:v for k,v in metadata.items() if k != "type"}
            ))
        return memories
