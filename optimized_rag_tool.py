import chromadb
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import re
import time

load_dotenv()

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant knowledge in the specified ChromaDB collection.")
    n_results: int = Field(default=5, description="Number of relevant documents to retrieve (increased from 3 for better coverage).")
    collection_name: str = Field(..., description="Name of the ChromaDB collection to query.")

class OptimizedRAGTool(BaseTool):
    name: str = "Optimized ChromaDB RAG Tool"
    description: str = (
        "An optimized Retrieval-Augmented Generation (RAG) tool with hybrid retrieval for faster and more accurate results. "
        "Features: 1) Semantic + Keyword hybrid search, 2) Query expansion, 3) Result reranking, 4) Caching for speed."
    )
    args_schema: type[RAGToolInput] = RAGToolInput

    def __init__(self, collection_name: str, db_path: str = "chroma.sqlite3"):
        super().__init__()
        persist_dir = os.path.dirname(os.path.abspath(db_path)) or "."
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        # 安全设置属性
        object.__setattr__(self, "_chroma_client", chromadb.PersistentClient(path=persist_dir))
        
        # 使用更快的嵌入模型
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"  # 更快更便宜
        )
        object.__setattr__(self, "_embedding_function", openai_ef)

        # 初始化collection
        object.__setattr__(self, "collection_name", collection_name)
        try:
            object.__setattr__(self, "_collection", self._chroma_client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            ))
        except Exception as e:
            print(f"Warning: Could not get collection '{collection_name}': {e}")
            object.__setattr__(self, "_collection", None)

        # 初始化缓存
        object.__setattr__(self, "_query_cache", {})
        object.__setattr__(self, "_keyword_cache", {})

    def _expand_query(self, query: str) -> List[str]:
        """扩展查询以提高检索效果"""
        if query in self._keyword_cache:
            return self._keyword_cache[query]
        
        expanded_queries = [query]
        
        # 添加常见问题模式的同义词
        question_patterns = {
            'what': ['which', 'define', 'explain'],
            'who': ['person', 'individual', 'people'],
            'when': ['time', 'date', 'year'],
            'where': ['location', 'place'],
            'how': ['method', 'way', 'process'],
            'why': ['reason', 'cause', 'purpose']
        }
        
        query_lower = query.lower()
        for pattern, synonyms in question_patterns.items():
            if pattern in query_lower:
                for synonym in synonyms[:2]:  # 限制数量以保持速度
                    expanded_queries.append(query.lower().replace(pattern, synonym))
        
        # 提取关键词
        keywords = self._extract_keywords(query)
        if keywords:
            expanded_queries.extend(keywords[:3])  # 只取前3个关键词
        
        # 缓存并限制数量
        self._keyword_cache[query] = expanded_queries[:4]
        return self._keyword_cache[query]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'what', 'which', 'who', 'when', 
                     'where', 'why', 'how'}
        
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]

    def _hybrid_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """混合搜索：语义搜索 + 关键词搜索"""
        if not self._collection:
            return {"documents": [], "metadatas": [], "distances": []}
        
        try:
            # 主要语义搜索
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results * 2, 10),  # 获取更多结果用于重排
                include=['documents', 'metadatas', 'distances']
            )
            
            # 如果结果不足，尝试扩展查询
            if not results.get("documents") or not results["documents"][0] or len(results["documents"][0]) < n_results:
                expanded_queries = self._expand_query(query)
                for exp_query in expanded_queries[1:3]:  # 尝试2个扩展查询
                    try:
                        exp_results = self._collection.query(
                            query_texts=[exp_query],
                            n_results=n_results // 2,
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        if exp_results.get("documents") and exp_results["documents"][0]:
                            # 简单合并结果
                            if results.get("documents") and results["documents"][0]:
                                results["documents"][0].extend(exp_results["documents"][0])
                                results["metadatas"][0].extend(exp_results["metadatas"][0]) 
                                results["distances"][0].extend(exp_results["distances"][0])
                            else:
                                results = exp_results
                            break
                    except:
                        continue
            
            # 去重并取前n_results个
            if results.get("documents") and results["documents"][0]:
                seen = set()
                unique_docs = []
                unique_metas = []
                unique_distances = []
                
                for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                    if doc not in seen:
                        seen.add(doc)
                        unique_docs.append(doc)
                        unique_metas.append(meta)
                        unique_distances.append(dist)
                        if len(unique_docs) >= n_results:
                            break
                
                results = {
                    "documents": [unique_docs],
                    "metadatas": [unique_metas],
                    "distances": [unique_distances]
                }
            
            return results
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            # 回退到基础搜索
            try:
                return self._collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
            except:
                return {"documents": [], "metadatas": [], "distances": []}

    def _format_results_with_context(self, results: Dict[str, Any], query: str) -> List[str]:
        """增强的结果格式化，添加权重评分"""
        docs_with_metadata = []
        
        if not results.get("documents") or not results["documents"][0]:
            return ["❌ No relevant documents found in the knowledge base."]
        
        query_keywords = set(self._extract_keywords(query))
        
        # 为结果添加权重评分并排序
        scored_results = []
        for i in range(len(results["documents"][0])):
            doc_content = results["documents"][0][i]
            doc_metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - distance
            
            # 计算关键词匹配加分
            content_keywords = set(self._extract_keywords(doc_content.lower()))
            keyword_matches = len(query_keywords.intersection(content_keywords))
            
            # 综合评分：语义相似度 + 关键词匹配
            total_score = similarity + (keyword_matches * 0.1)
            
            scored_results.append((total_score, doc_content, doc_metadata, similarity))
        
        # 按总分排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 格式化排序后的结果
        for i, (score, doc_content, doc_metadata, similarity) in enumerate(scored_results):
            relevance = "🟢 High" if similarity > 0.8 else "🟡 Medium" if similarity > 0.6 else "🔴 Low"
            
            # 检查关键词匹配
            content_keywords = set(self._extract_keywords(doc_content.lower()))
            matching_keywords = query_keywords.intersection(content_keywords)
            keyword_info = f" [Keywords: {', '.join(matching_keywords)}]" if matching_keywords else ""
            
            # 提取元数据
            source_info = doc_metadata.get('source', 'Unknown')
            page_info = doc_metadata.get('page', 'N/A')
            doc_type = doc_metadata.get('type', 'content')
            
            # 格式化结果（排名第一的结果更突出）
            rank_indicator = "🏆" if i == 0 else f"{i+1}."
            
            formatted_result = (
                f"📄 **{rank_indicator} Content:** {doc_content}\n"
                f"🔍 **Relevance:** {relevance} ({similarity:.3f}) Score: {score:.3f}\n"
                f"📁 **Source:** {source_info}\n"
                f"📖 **Page:** {page_info}\n"
                f"🏷️ **Type:** {doc_type}"
                f"{keyword_info}\n"
                f"{'─' * 50}"
            )
            
            docs_with_metadata.append(formatted_result)
        
        return docs_with_metadata

    def _run(self, query: str, n_results: int = 5, collection_name: str = None) -> List[str]:
        """增强的查询执行 - 添加详细调试信息"""
        start_time = time.time()
        
        # 添加详细调试输出
        print(f"\n🔍 RAG DEBUG INFO:")
        print(f"   📝 Query: '{query[:100]}{'...' if len(query) > 100 else ''}' ({len(query)} chars)")
        print(f"   🎯 Searching for {n_results} results in collection: {self.collection_name}")
        
        # 缓存检查
        cache_key = f"{query}_{n_results}_{collection_name or self.collection_name}"
        if cache_key in self._query_cache:
            print(f"   ⚡ Cache hit! Time: {time.time() - start_time:.3f}s")
            return self._query_cache[cache_key]
        
        if not self._collection:
            error_msg = ["❌ Error: No collection available for query."]
            print(f"   ❌ No collection available!")
            self._query_cache[cache_key] = error_msg
            return error_msg

        try:
            # 执行混合搜索
            results = self._hybrid_search(query, n_results)
            
            # 调试：显示原始搜索结果
            if results.get("documents") and results["documents"][0]:
                print(f"   ✅ Found {len(results['documents'][0])} raw results")
                # 显示最佳结果的详细信息
                best_doc = results["documents"][0][0]
                best_similarity = 1 - results["distances"][0][0]
                best_source = results["metadatas"][0][0].get('source', 'Unknown')
                best_page = results["metadatas"][0][0].get('page', 'N/A')
                
                print(f"   🎯 Best match:")
                print(f"      📊 Similarity: {best_similarity:.3f}")
                print(f"      📄 Content: '{best_doc[:150]}{'...' if len(best_doc) > 150 else ''}'")
                print(f"      📁 Source: {best_source} (Page: {best_page})")
                
                # 显示关键词匹配情况
                query_keywords = set(self._extract_keywords(query))
                content_keywords = set(self._extract_keywords(best_doc.lower()))
                matching_keywords = query_keywords.intersection(content_keywords)
                if matching_keywords:
                    print(f"      🔑 Matching keywords: {', '.join(matching_keywords)}")
                else:
                    print(f"      🔑 No exact keyword matches (semantic similarity)")
                
            else:
                print(f"   ❌ No results found!")
                print(f"   💡 Try:")
                print(f"      - Rephrasing the question")
                print(f"      - Using different keywords")
                print(f"      - Checking if the information exists in the collection")
            
            # 格式化结果
            formatted_results = self._format_results_with_context(results, query)
            
            # 缓存结果
            self._query_cache[cache_key] = formatted_results
            
            query_time = time.time() - start_time
            print(f"   ⏱️ Total RAG time: {query_time:.3f}s")
            print(f"   📦 Returning {len(formatted_results)} formatted results")
            
            return formatted_results
            
        except Exception as e:
            import traceback
            query_time = time.time() - start_time
            print(f"   ❌ RAG Error: {e}")
            print(f"   📍 Error location: {traceback.format_exc().split('File')[-1] if traceback.format_exc() else 'Unknown'}")
            
            error_results = [
                f"❌ Error querying ChromaDB: {e}",
                f"Query time: {query_time:.3f}s",
                f"Please check collection and try again."
            ]
            
            # 缓存错误以防重复
            self._query_cache[cache_key] = error_results
            return error_results

    def clear_cache(self):
        """清除缓存"""
        old_query_size = len(self._query_cache)
        old_keyword_size = len(self._keyword_cache)
        
        self._query_cache.clear()
        self._keyword_cache.clear()
        
        print(f"🧹 Cache cleared: {old_query_size} queries, {old_keyword_size} keywords")

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        return {
            "query_cache_size": len(self._query_cache),
            "keyword_cache_size": len(self._keyword_cache),
            "collection_name": self.collection_name
        }

    def debug_query(self, query: str) -> Dict[str, Any]:
        """专门用于调试的查询方法"""
        print(f"\n🔬 DETAILED DEBUG FOR QUERY: '{query}'")
        print("="*60)
        
        if not self._collection:
            return {"error": "No collection available"}
        
        try:
            # 1. 显示原始查询分析
            keywords = self._extract_keywords(query)
            expanded = self._expand_query(query)
            print(f"📝 Original query: '{query}'")
            print(f"🔑 Extracted keywords: {keywords}")
            print(f"📈 Expanded queries: {expanded}")
            
            # 2. 执行搜索并显示详细结果
            results = self._collection.query(
                query_texts=[query],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results.get("documents") and results["documents"][0]:
                print(f"\n📊 Found {len(results['documents'][0])} results:")
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0][:5], 
                    results["metadatas"][0][:5], 
                    results["distances"][0][:5]
                )):
                    similarity = 1 - dist
                    print(f"\n{i+1}. Similarity: {similarity:.3f}")
                    print(f"   Content: '{doc[:200]}...'")
                    print(f"   Source: {meta.get('source', 'Unknown')} (Page: {meta.get('page', 'N/A')})")
                    
                    # 检查关键词匹配
                    content_keywords = set(self._extract_keywords(doc.lower()))
                    query_keywords = set(keywords)
                    matches = query_keywords.intersection(content_keywords)
                    print(f"   Keywords: {', '.join(matches) if matches else 'No exact matches'}")
            
            return {"status": "success", "results_count": len(results["documents"][0]) if results.get("documents") else 0}
            
        except Exception as e:
            print(f"❌ Debug error: {e}")
            return {"error": str(e)}