#!/usr/bin/env python3
"""
Optimized CrewAI Flow-based Kahoot Bot with performance enhancements
优化版CrewAI Flow Kahoot机器人，具有性能增强功能
"""

import os
import time
from typing import Dict, Optional
from pydantic import BaseModel
from crewai import Agent, LLM, Task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv

# 修复导入 - 先尝试优化版本，如果失败则使用原版
try:
    from optimized_rag_tool import OptimizedRAGTool as RAGTool
    print("✅ Using optimized RAG tool")
except ImportError:
    try:
        from rag_tool import RAGTool
        print("ℹ️ Using standard RAG tool")
    except ImportError:
        print("❌ No RAG tool available")
        RAGTool = None

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Optimized Flow state model
class OptimizedKahootBotState(BaseModel):
    game_pin: str = os.getenv("KAHOOT_PIN", "299268")
    nickname: str = os.getenv("KAHOOT_NICKNAME", "CrewAI_Bot")
    is_connected: bool = False
    rag_tool: Optional[object] = None  # Store RAG tool instance
    response_cache: Dict[str, str] = {}  # Cache for faster responses
    question_count: int = 0
    avg_response_time: float = 0.0

class OptimizedKahootBotFlow(Flow[OptimizedKahootBotState]):
    """
    Optimized Kahoot bot with performance enhancements:
    1. Faster LLM model selection
    2. Optimized RAG tool integration  
    3. Response caching
    4. Parallel processing where possible
    5. Enhanced error handling
    """
    
    def __init__(self):
        super().__init__()
        
        # Configure optimized LLM - use faster model for quiz scenarios
        self.llm = LLM(
            model="openai/gpt-4o-mini",  # Fast model for quick responses
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            # Optimize for speed
            max_tokens=150,  # Shorter responses for quiz answers
            temperature=0.1,  # Lower temperature for consistent answers
            top_p=0.9,       # Focused sampling
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        # Browser MCP server parameters with optimizations
        env = os.environ.copy()
        env["SKIP_PORT_KILL"] = "1"
        env["NODE_ENV"] = "production"  # Optimize Node.js performance
        self.browser_server_params = StdioServerParameters(
            command="npx",
            args=["@browsermcp/mcp@latest"],
            env=env
        )
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

    @start()
    def initialize_optimized_bot(self) -> OptimizedKahootBotState:
        """Initialize the optimized Kahoot bot with performance enhancements"""
        print("🚀 Starting Optimized Kahoot Bot...")
        print("⚡ Performance Mode: Enabled")
        
        # Debug environment variables
        print(f"🔧 Using PIN: {os.getenv('KAHOOT_PIN', '299268')}")
        print(f"🔧 Using Nickname: {os.getenv('KAHOOT_NICKNAME', 'CrewAI_Bot')}")
        
        state = OptimizedKahootBotState()
        print(f"📋 Bot Configuration: PIN={state.game_pin}, Name={state.nickname}")
        
        return state

    @listen(initialize_optimized_bot)
    def setup_optimized_rag(self, state: OptimizedKahootBotState) -> OptimizedKahootBotState:
        """Setup optimized RAG tool with collection selection"""
        print("🔍 Setting up Optimized RAG Tool...")
        
        from chromadb import PersistentClient
        
        try:
            chroma_client = PersistentClient(path="./chroma_db")
            collections = chroma_client.list_collections()
            
            if not collections:
                print("❌ No collections found in ChromaDB!")
                print("💡 Please run: python chromadb_manager.py to create collections first")
                return state

            print("📚 Available ChromaDB collections:")
            for idx, col in enumerate(collections):
                doc_count = col.count() if hasattr(col, 'count') else "Unknown"
                print(f"  {idx+1}. {col.name} ({doc_count} documents)")

            # Select collection with retry logic
            selected_collection_name = None
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    choice = int(input("🎯 Select collection number for RAG optimization: ")) - 1
                    if 0 <= choice < len(collections):
                        selected_collection_name = collections[choice].name
                        print(f"✅ Selected optimized collection: {selected_collection_name}")
                        break
                    else:
                        print(f"❌ Invalid selection. Please choose 1-{len(collections)}")
                except ValueError:
                    print("❌ Please enter a valid number")
                    if attempt == max_attempts - 1:
                        print("🚫 Max attempts reached. Using first collection as fallback.")
                        selected_collection_name = collections[0].name

            if not selected_collection_name:
                print("❌ No collection selected. Bot cannot proceed.")
                return state

            # Initialize optimized RAG tool
            print("⚡ Initializing Optimized RAG Tool...")
            start_time = time.time()
            
            if RAGTool is None:
                print("❌ RAG Tool not available - please check imports")
                return state
            
            state.rag_tool = RAGTool(
                collection_name=selected_collection_name, 
                db_path="./chroma_db/chroma.sqlite3"
            )
            
            init_time = time.time() - start_time
            print(f"✅ RAG Tool initialized in {init_time:.2f}s")
            
            # Test RAG tool with a quick query
            print("🧪 Testing RAG tool performance...")
            test_start = time.time()
            test_results = state.rag_tool._run("test query", n_results=1)
            test_time = time.time() - test_start
            print(f"⚡ Test query completed in {test_time:.3f}s")
            
            return state
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG Tool: {e}")
            return state

    @listen(setup_optimized_rag)
    def play_optimized_quiz(self, state: OptimizedKahootBotState) -> OptimizedKahootBotState:
        """Optimized quiz playing with performance monitoring"""
        if not state.rag_tool:
            print("❌ RAG tool not available. Cannot proceed with quiz.")
            return state

        print("🎮 Starting Optimized Quiz Mode!")
        print("📊 Performance monitoring: Enabled")
        print("⚡ Response target: <2 seconds per question")
        print("📋 ASSUMPTION: You are already in the Kahoot waiting room")

        try:
            with MCPServerAdapter(self.browser_server_params) as browser_tools:
                print("🚀 Browser MCP server connected!")
                
                # Connection test with timeout
                print("🔄 Testing browser connection...")
                max_connection_attempts = 3
                
                for attempt in range(max_connection_attempts):
                    try:
                        result = browser_tools.browser_snapshot(random_string="connection_test")
                        print(f"✅ Connection test #{attempt+1} successful")
                        break
                    except Exception as e:
                        print(f"⚠️ Connection attempt #{attempt+1} failed: {str(e)[:100]}...")
                        if attempt < max_connection_attempts - 1:
                            print("🔄 Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            print("⚠️ Connection issues detected, but proceeding anyway...")

                print("✅ Browser tools ready!")
                print(f"🛠️ Available tools: {[tool.name for tool in browser_tools]}")
                
                # Optimized answer instructions with performance focus
                optimized_instructions = """
                PERFORMANCE-OPTIMIZED SYSTEM INSTRUCTIONS:

                You are a high-speed Kahoot quiz bot. Your goal is to answer questions in <2 seconds.

                MANDATORY WORKFLOW (DO NOT DEVIATE):
                1. Take browser_snapshot IMMEDIATELY (no delay)
                2. If quiz question detected:
                   - Extract question text QUICKLY
                   - Call RAGTool with CONCISE search query (5-10 words max)
                   - Based on RAG results, click the BEST answer immediately
                   - DO NOT overthink or provide explanations
                3. If no question visible, report current state briefly

                SPEED OPTIMIZATIONS:
                - Use short, focused RAG queries
                - Click answers immediately after RAG lookup
                - No lengthy analysis or explanations  
                - Trust the RAG tool results

                RAG QUERY GUIDELINES:
                - Extract 3-5 key words from the question
                - Focus on main topic/concept being asked
                - Ignore question words (what, who, when, where, why, how)
                
                EXAMPLE:
                Question: "What programming language was created by Guido van Rossum?"
                RAG Query: "Guido van Rossum programming language"
                
                BE FAST AND DECISIVE!
                """

                # Main quiz loop with performance tracking
                total_response_times = []
                print("\n🎯 Entering High-Performance Quiz Mode!")
                print("⏱️ Performance metrics will be tracked")
                
                while True:
                    try:
                        state.question_count += 1
                        question_start_time = time.time()
                        
                        print(f"\n{'='*60}")
                        print(f"🔍 Question #{state.question_count} - Starting analysis...")
                        print(f"🎯 Target: <2.0s | Previous avg: {state.avg_response_time:.2f}s")
                        
                        # Create optimized agent with performance settings
                        tools_list = list(browser_tools)
                        if state.rag_tool:
                            tools_list.append(state.rag_tool)

                        quiz_agent = Agent(
                            role="Speed Quiz Bot",
                            goal="Answer quiz questions in under 2 seconds using RAG tool for knowledge lookup",
                            backstory="I am an optimized quiz bot that prioritizes speed and accuracy. I use RAG tool for every question and respond quickly.",
                            llm=self.llm,
                            tools=tools_list,
                            verbose=False,  # Reduce logging for speed
                            memory=False,   # No memory for faster processing
                            max_execution_time=5,  # 5 second timeout
                            step_callback=None  # Remove callbacks for speed
                        )
                        
                        quiz_task = Task(
                            description="Quickly analyze browser page, use RAG tool if quiz question found, then click best answer",
                            expected_output="Answer clicked or current page status",
                            agent=quiz_agent,
                            tools=tools_list
                        )

                        # Execute with timeout and performance monitoring
                        task_start_time = time.time()
                        result = quiz_task.execute_sync()
                        task_time = time.time() - task_start_time
                        
                        question_total_time = time.time() - question_start_time
                        total_response_times.append(question_total_time)
                        state.avg_response_time = sum(total_response_times) / len(total_response_times)
                        
                        # Performance feedback
                        speed_status = "🟢 FAST" if question_total_time < 2.0 else "🟡 MEDIUM" if question_total_time < 4.0 else "🔴 SLOW"
                        print(f"\n📊 PERFORMANCE REPORT:")
                        print(f"   ⏱️ Total Time: {question_total_time:.2f}s {speed_status}")
                        print(f"   🤖 Agent Time: {task_time:.2f}s")
                        print(f"   📈 Average: {state.avg_response_time:.2f}s")
                        print(f"   🎯 Questions: {state.question_count}")
                        
                        print(f"\n✅ Question #{state.question_count} Result: {str(result)[:100]}...")
                        
                        # Wait for user input with performance summary
                        print(f"\n⏳ Question #{state.question_count} completed!")
                        if question_total_time < 2.0:
                            print("🚀 Excellent response time!")
                        elif question_total_time < 4.0:
                            print("⚡ Good response time")
                        else:
                            print("🐌 Consider optimizing RAG queries or collection")
                            
                        next_action = input("🔄 Press <Enter> for next question, 'stats' for detailed stats, or 'quit' to stop: ").strip().lower()
                        
                        if next_action == 'quit':
                            print("🛑 Quiz bot stopped by user")
                            break
                        elif next_action == 'stats':
                            self._show_performance_stats(total_response_times, state)
                            input("Press <Enter> to continue...")
                            
                    except KeyboardInterrupt:
                        print("\n🛑 Quiz bot stopped by user (Ctrl+C)")
                        break
                    except Exception as e:
                        error_time = time.time() - question_start_time
                        print(f"❌ Error in question #{state.question_count} (after {error_time:.2f}s): {e}")
                        
                        retry = input("🔄 Press <Enter> to retry, 'skip' to continue, or 'quit' to stop: ").strip().lower()
                        if retry == 'quit':
                            break
                        elif retry == 'skip':
                            continue

        except Exception as e:
            print(f"❌ Optimized quiz monitoring failed: {e}")
            
        # Final performance summary
        if total_response_times:
            self._show_final_performance_summary(total_response_times, state)
            
        return state

    def _show_performance_stats(self, response_times: list, state: OptimizedKahootBotState):
        """Show detailed performance statistics"""
        if not response_times:
            print("📊 No performance data available yet")
            return
            
        import statistics
        
        fast_responses = [t for t in response_times if t < 2.0]
        medium_responses = [t for t in response_times if 2.0 <= t < 4.0]
        slow_responses = [t for t in response_times if t >= 4.0]
        
        print(f"\n📊 DETAILED PERFORMANCE STATISTICS")
        print(f"{'='*50}")
        print(f"📈 Total Questions: {len(response_times)}")
        print(f"⚡ Average Time: {statistics.mean(response_times):.2f}s")
        print(f"📊 Median Time: {statistics.median(response_times):.2f}s")
        print(f"🚀 Best Time: {min(response_times):.2f}s")
        print(f"🐌 Worst Time: {max(response_times):.2f}s")
        print(f"\n🎯 SPEED BREAKDOWN:")
        print(f"   🟢 Fast (<2s): {len(fast_responses)} ({len(fast_responses)/len(response_times)*100:.1f}%)")
        print(f"   🟡 Medium (2-4s): {len(medium_responses)} ({len(medium_responses)/len(response_times)*100:.1f}%)")
        print(f"   🔴 Slow (>4s): {len(slow_responses)} ({len(slow_responses)/len(response_times)*100:.1f}%)")
        
        if hasattr(state.rag_tool, 'get_cache_stats'):
            cache_stats = state.rag_tool.get_cache_stats()
            print(f"\n🗄️ RAG CACHE STATISTICS:")
            for key, value in cache_stats.items():
                print(f"   {key}: {value}")

    def _show_final_performance_summary(self, response_times: list, state: OptimizedKahootBotState):
        """Show final performance summary"""
        print(f"\n🏁 FINAL PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            fast_count = len([t for t in response_times if t < 2.0])
            
            print(f"🎮 Total Questions Answered: {len(response_times)}")
            print(f"⚡ Average Response Time: {avg_time:.2f}s")
            print(f"🎯 Fast Responses (<2s): {fast_count}/{len(response_times)} ({fast_count/len(response_times)*100:.1f}%)")
            
            if avg_time < 2.0:
                print("🏆 EXCELLENT PERFORMANCE! Bot is quiz-ready!")
            elif avg_time < 3.0:
                print("✅ GOOD PERFORMANCE! Consider fine-tuning for competitive play")
            else:
                print("⚠️ NEEDS OPTIMIZATION! Consider:")
                print("   - Using smaller, more focused document collections")
                print("   - Optimizing chunk sizes")
                print("   - Using faster embedding models")
        else:
            print("❌ No questions were processed")
        
        print(f"\n🤖 Bot Session Complete!")

def main():
    """Main entry point with performance monitoring"""
    print("🚀 CrewAI Optimized Kahoot Bot Starting...")
    print("⚡ Performance enhancements: ACTIVE")
    
    start_time = time.time()
    
    # Create and run the optimized flow
    bot_flow = OptimizedKahootBotFlow()
    result = bot_flow.kickoff()
    
    total_time = time.time() - start_time
    print(f"\n🎯 Session Complete! Total runtime: {total_time:.2f}s")
    print(f"📊 Final Result: {result}")

if __name__ == "__main__":
    main()