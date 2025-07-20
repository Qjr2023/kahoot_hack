"""
Fixed Optimized ChromaDB Manager - 基于原版的优化
直接复制这个代码替换 optimized_chromadb_manager.py
"""

import os
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OptimizedDoclingChromaProcessor:
    """
    优化的处理器，保持与原版兼容的接口
    """
    
    def __init__(self,
                 openai_api_key: str = None,
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "text-embedding-3-small"):  # 更快的嵌入模型
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.chroma_db_path = chroma_db_path
        self.embedding_model = embedding_model
        
        # 使用原版的初始化方式，只是优化参数
        self.converter = self._setup_docling_converter()
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.embedding_function = self._setup_embedding_function()
        
        # 添加缓存
        self.document_cache = {}

    def _setup_docling_converter(self) -> DocumentConverter:
        """与原版相同的设置方式"""
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
        )
        
        return converter

    def _setup_embedding_function(self):
        """优化的嵌入函数 - 使用更快的模型"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model  # 这是主要优化点
        )

    def extract_text_with_docling(self, file_path: str) -> Dict[str, Any]:
        """修复版本兼容性问题"""
        # 添加简单的缓存
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.document_cache:
            print(f"⚡ Using cached extraction for {Path(file_path).name}")
            return self.document_cache[file_hash]
        
        try:
            start_time = time.time()
            result = self.converter.convert(file_path)
            processing_time = time.time() - start_time

            # 修复: 只获取markdown内容，避免版本兼容问题
            markdown_content = result.document.export_to_markdown()
            
            # 尝试获取JSON，如果失败就跳过
            json_content = None
            try:
                if hasattr(result.document, 'export_to_json'):
                    json_content = result.document.export_to_json()
                elif hasattr(result.document, 'to_json'):
                    json_content = result.document.to_json()
                else:
                    print("ℹ️ JSON export not available in this Docling version")
            except Exception as json_error:
                print(f"ℹ️ JSON export failed: {json_error}")
                json_content = None

            # 安全获取title
            title = Path(file_path).name  # 默认值
            if hasattr(result.document, 'title') and result.document.title:
                title = result.document.title
            elif hasattr(result, 'title') and result.title:
                title = result.title

            metadata = {
                "source": file_path,
                "title": title,
                "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1,
                "format": "markdown",
                "tables_count": self._count_tables(result.document),
                "images_count": self._count_images(result.document),
                "processing_time": processing_time,
                "word_count": len(markdown_content.split()),
                "char_count": len(markdown_content)
            }

            extracted_data = {
                "content": markdown_content,
                "json_content": json_content,  # 可能为None
                "metadata": metadata,
                "document_object": result.document
            }
            
            # 缓存结果
            self.document_cache[file_hash] = extracted_data
            print(f"✅ Document extracted in {processing_time:.2f}s ({metadata['word_count']} words)")
            return extracted_data

        except Exception as e:
            print(f"❌ Docling parsing failed for '{file_path}': {e}")
            print(f"💡 Please check if the file is a valid PDF and try again")
            return None

    def _get_file_hash(self, file_path: str) -> str:
        """生成文件哈希用于缓存"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _count_tables(self, document) -> int:
        """兼容版本的表格计数"""
        try:
            if hasattr(document, 'body') and hasattr(document.body, 'items'):
                return len([item for item in document.body.items if hasattr(item, 'label') and item.label == 'table'])
            else:
                # 尝试其他可能的属性名
                return 0
        except Exception as e:
            print(f"ℹ️ Cannot count tables: {e}")
            return 0
    
    def _count_images(self, document) -> int:
        """兼容版本的图片计数"""
        try:
            if hasattr(document, 'body') and hasattr(document.body, 'items'):
                return len([item for item in document.body.items if hasattr(item, 'label') and item.label == 'picture'])
            else:
                return 0
        except Exception as e:
            print(f"ℹ️ Cannot count images: {e}")
            return 0

    def create_or_get_collection(self, collection_name: str):
        """与原版相同的接口"""
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Collection '{collection_name}' is ready.")
        return collection
    
    def chunk_content(self, content: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """优化的分块 - 稍微小一点的chunk和更多overlap"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            if end < len(content):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = content[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def embed_document(self, collection_name: str, file_path: str,
                      chunk_size: int = 800, overlap: int = 150) -> bool:
        """与原版兼容，但使用优化参数"""
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}")
            return False

        collection = self.create_or_get_collection(collection_name)

        existing_docs = collection.get(where={"source": file_path})
        if existing_docs['ids']:
            print(f"ℹ️ Document '{file_path}' already exists. Skipping.")
            return False

        extracted_data = self.extract_text_with_docling(file_path)
        if not extracted_data:
            return False

        content = extracted_data['content']
        metadata = extracted_data['metadata']

        chunks = self.chunk_content(content, chunk_size, overlap)

        documents = chunks
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            metadatas.append(chunk_metadata)

        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"✅ Successfully embedded {len(chunks)} chunks from '{file_path}'.")
            return True
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return False

# 复用原版的交互界面类
class InteractiveKnowledgeBase:
    def __init__(self):
        # 使用优化的处理器，但保持相同接口
        self.processor = OptimizedDoclingChromaProcessor()
        self.current_collection = None
    
    def run(self):
        """完全相同的交互界面"""
        print("=== 优化版 Docling + ChromaDB Knowledge Base Manager ===")
        print("🚀 性能优化: 启用更快的嵌入模型和缓存")
        
        while True:
            print("\nOptions:")
            print("1. List collections")
            print("2. Create/Select collection")
            print("3. Add document")
            print("4. Query collection")
            print("5. View collection stats")
            print("6. Delete collection")
            print("7. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "1":
                self.list_collections()
            elif choice == "2":
                self.create_or_select_collection()
            elif choice == "3":
                self.add_document()
            elif choice == "4":
                self.query_collection()
            elif choice == "5":
                self.view_stats()
            elif choice == "6":
                self.delete_collection()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option")
    
    def list_collections(self):
        collections = self.processor.chroma_client.list_collections()
        if collections:
            print("\n📚 Available collections:")
            for i, collection in enumerate(collections, 1):
                try:
                    count = collection.count()
                    print(f"{i}. {collection.name} ({count} documents)")
                except:
                    print(f"{i}. {collection.name}")
        else:
            print("No collections found")
    
    def create_or_select_collection(self):
        name = input("Enter collection name: ").strip()
        if name:
            try:
                self.processor.create_or_get_collection(name)
                self.current_collection = name
                print(f"Selected collection: {name}")
            except Exception as e:
                print(f"Error: {e}")
    
    def add_document(self):
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        file_path = input("Enter file path: ").strip()
        if os.path.isfile(file_path):
            print(f"🚀 Processing with optimizations...")
            success = self.processor.embed_document(
                collection_name=self.current_collection,
                file_path=file_path
            )
            if success:
                print("✅ Document added successfully with optimizations!")
            else:
                print("❌ Failed to add document")
        else:
            print("File not found")
    
    def query_collection(self):
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        # 这里暂时使用简单查询，可以后续优化
        query = input("Enter your query: ").strip()
        if query:
            try:
                collection = self.processor.chroma_client.get_collection(
                    name=self.current_collection,
                    embedding_function=self.processor.embedding_function
                )
                
                results = collection.query(
                    query_texts=[query],
                    n_results=5  # 优化: 从3改为5
                )
                
                print(f"\nFound {len(results['documents'][0])} results:")
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                    print(f"\n--- Result {i} ---")
                    print(f"Content: {doc[:300]}...")
                    print(f"Source: {meta.get('source', 'unknown')}")
                    
            except Exception as e:
                print(f"Query failed: {e}")
    
    def view_stats(self):
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        try:
            collection = self.processor.chroma_client.get_collection(
                name=self.current_collection,
                embedding_function=self.processor.embedding_function
            )
            count = collection.count()
            print(f"\n📊 Collection '{self.current_collection}' Statistics:")
            print(f"Total documents: {count}")
            print(f"Embedding model: {self.processor.embedding_model}")
            print(f"Cache hits: {len(self.processor.document_cache)}")
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    def delete_collection(self):
        self.list_collections()
        name = input("Enter collection name to delete: ").strip()
        if name:
            confirm = input(f"Are you sure you want to delete '{name}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    self.processor.chroma_client.delete_collection(name)
                    print(f"Collection '{name}' deleted")
                    if self.current_collection == name:
                        self.current_collection = None
                except Exception as e:
                    print(f"Error: {e}")

# 重要: 必须有这个主程序入口!
if __name__ == "__main__":
    print("🚀 Starting Optimized ChromaDB Manager...")
    manager = InteractiveKnowledgeBase()
    manager.run()