#!/usr/bin/env python3
"""
Direct TXT/MD to ChromaDB importer
Bypasses Docling to avoid version compatibility issues
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def add_text_to_chromadb(file_path: str, collection_name: str, 
                        chroma_db_path: str = "./chroma_db"):
    """Add text file directly to ChromaDB"""
    
    print(f"Processing file: {file_path}")
    
    # Check file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"File read successfully. Content length: {len(content)} characters")
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    if len(content.strip()) < 100:
        print("Error: File content too short")
        return False
    
    # Initialize ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Setup embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Error: OPENAI_API_KEY not found")
            return False
        
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Collection '{collection_name}' ready")
        
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return False
    
    # Check if document already exists
    try:
        existing_docs = collection.get(where={"source": file_path})
        if existing_docs['ids']:
            print("Document already exists, skipping")
            return True
    except Exception as e:
        print(f"Warning: Error checking existing docs: {e}")
    
    # Chunk the text
    print("Starting text chunking...")
    chunks = chunk_text(content, chunk_size=1000, overlap=100)
    print(f"Chunking complete. Total chunks: {len(chunks)}")
    
    if not chunks:
        print("Error: Chunking failed")
        return False
    
    # Prepare data
    file_stem = Path(file_path).stem
    documents = chunks
    ids = [f"{file_stem}_direct_{i}" for i in range(len(chunks))]
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        metadata = {
            "source": file_path,
            "title": Path(file_path).name,
            "chunk_id": i,
            "chunk_size": len(chunk),
            "total_chunks": len(chunks),
            "format": "text",
            "extraction_method": "direct_text_import"
        }
        metadatas.append(metadata)
    
    # Add to ChromaDB in batches
    batch_size = 25
    total_added = 0
    
    print("Adding to ChromaDB...")
    
    try:
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas
            )
            
            total_added += len(batch_docs)
            print(f"Progress: {total_added}/{len(documents)} chunks added")
            
            # Brief pause
            import time
            time.sleep(0.1)
        
        print("Import complete!")
        print(f"  File: {file_path}")
        print(f"  Collection: {collection_name}")
        print(f"  Total chunks: {total_added}")
        return True
        
    except Exception as e:
        print(f"Error adding to ChromaDB: {e}")
        return False

def test_query(collection_name: str, query: str, chroma_db_path: str = "./chroma_db"):
    """Test query functionality"""
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-ada-002"
        )
        
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results['documents'][0])} results:")
        
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ), 1):
            print(f"\n--- Result {i} ---")
            print(f"Source: {meta.get('source', 'unknown')}")
            print(f"Similarity: {1-dist:.3f}")
            print(f"Content preview: {doc[:200]}...")
            
    except Exception as e:
        print(f"Query failed: {e}")

def main():
    """Main function"""
    print("Direct Text File to ChromaDB Importer")
    print("="*40)
    
    # Get input parameters
    txt_file = input("Enter text file path: ").strip().strip('"\'')
    if not txt_file:
        txt_file = "extracted_round3_pdfplumber.md"  # Default
    
    collection_name = input("Enter collection name (default: round3_docs): ").strip()
    if not collection_name:
        collection_name = "round3_docs"
    
    # Import file
    success = add_text_to_chromadb(txt_file, collection_name)
    
    if success:
        print("\nImport successful!")
        
        # Test query
        test_query_input = input("\nEnter test query (or press Enter to skip): ").strip()
        if test_query_input:
            test_query(collection_name, test_query_input)
        
        print(f"\nYou can now use collection '{collection_name}' in kahoot_bot.py")
        
    else:
        print("\nImport failed")

if __name__ == "__main__":
    main()