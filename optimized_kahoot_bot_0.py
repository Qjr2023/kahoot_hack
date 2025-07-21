#!/usr/bin/env python3
"""
Kahoot Bot with Fixed Accuracy Issues
Main fixes: Question extraction, RAG filtering, LLM fallback strategies
"""

import os
import time
import re
import requests
import json
from typing import List, Dict
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv

load_dotenv()

class FixedAccuracyKahootBot:
    """Kahoot Bot with Fixed Accuracy Issues"""
    
    def __init__(self):
        # LLM Configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Browser MCP Configuration
        env = os.environ.copy()
        env["SKIP_PORT_KILL"] = "1"
        self.browser_server_params = StdioServerParameters(
            command="npx",
            args=["@browsermcp/mcp@latest"],
            env=env
        )
        
        # RAG Tool (optional)
        self.rag_tool = None
        self.setup_rag()
    
    def setup_rag(self):
        """Setup RAG tool"""
        try:
            from chromadb import PersistentClient
            from rag_tool import RAGTool
            
            chroma_client = PersistentClient(path="./chroma_db")
            collections = chroma_client.list_collections()
            
            if collections:
                print("📚 Available collections:")
                for idx, col in enumerate(collections):
                    print(f"{idx+1}. {col.name}")
                
                choice = int(input("Select collection (or 0 to skip RAG): ")) - 1
                if choice >= 0 and choice < len(collections):
                    selected = collections[choice].name
                    self.rag_tool = RAGTool(
                        collection_name=selected,
                        db_path="./chroma_db/chroma.sqlite3"
                    )
                    print(f"✅ RAG enabled: {selected}")
                else:
                    print("⚠️ RAG disabled")
            else:
                print("⚠️ No collections found, RAG disabled")
        except Exception as e:
            print(f"⚠️ RAG setup failed: {e}")
    
    def extract_question_and_options(self, page_content: str) -> Dict:
        """Improved question and options extraction"""
        try:
            # More comprehensive question extraction patterns
            question_patterns = [
                r'heading "([^"]*)"[^>]*\[level=1\]',  # Main heading
                r'<h1[^>]*>([^<]+)</h1>',
                r'"question[^"]*"[^>]*>([^<]+)<',
                r'- heading "([^"]*)".*?\[level=1\]',
                # Handle truncated questions
                r'heading "([^"]*\\[^"]*)"',  # Truncated questions with backslashes
            ]
            
            question = "Unknown question"
            for pattern in question_patterns:
                matches = re.findall(pattern, page_content, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Choose the longest match (usually the complete question)
                    question = max(matches, key=len).strip()
                    break
            
            # Clean question text
            question = re.sub(r'\\+$', '', question)  # Remove trailing backslashes
            question = question.replace('\\n', ' ').replace('\\', '').strip()
            
            # Find option buttons - more precise patterns
            button_patterns = [
                r'- button "([^"]+)" \[ref=([^\]]+)\]',
                r'button[^>]*>([^<]+)</button>.*?ref["\']([^"\']+)["\']',
            ]
            
            buttons = []
            for pattern in button_patterns:
                buttons = re.findall(pattern, page_content)
                if buttons:
                    break
            
            # Filter and clean options
            options = []
            refs = []
            for button in buttons[:4]:  # Maximum 4 options
                option_text = button[0].strip()
                option_ref = button[1].strip()
                
                # Skip buttons that are obviously not answers
                if len(option_text) > 2 and option_text not in ['Next', 'Skip', 'Submit']:
                    options.append(option_text)
                    refs.append(option_ref)
            
            return {
                "question": question,
                "options": options,
                "refs": refs,
                "valid": len(options) >= 2 and len(question) > 10  # Ensure question is complete
            }
        except Exception as e:
            print(f"❌ Extract failed: {e}")
            return {"question": "", "options": [], "refs": [], "valid": False}
    
    def get_smart_rag_context(self, question: str) -> str:
        """Smart RAG query - only return content when relevant"""
        if not self.rag_tool:
            return ""
        
        try:
            rag_result = self.rag_tool._run(query=question, n_results=2)
            
            if not rag_result or len(rag_result) == 0:
                return ""
            
            # Get RAG content
            rag_content = str(rag_result[0])
            
            # Check if RAG content is relevant to the question
            question_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
            rag_keywords = set(re.findall(r'\b\w{4,}\b', rag_content.lower()))
            
            # Calculate keyword overlap
            overlap = len(question_keywords.intersection(rag_keywords))
            relevance_score = overlap / max(len(question_keywords), 1)
            
            print(f"🔍 RAG relevance score: {relevance_score:.2f}")
            
            # Only use RAG content when relevance is high enough
            if relevance_score > 0.6:  # Raised to 60% keyword overlap
                print(f"✅ RAG content is relevant (score: {relevance_score:.2f})")
                return rag_content[:400]  # Limit length
            else:
                print(f"❌ RAG content not relevant enough (score: {relevance_score:.2f}), using pure LLM")
                return ""
                
        except Exception as e:
            print(f"⚠️ RAG query failed: {e}")
            return ""
    
    def direct_llm_call(self, question: str, options: List[str], rag_context: str = "", timeout: int = 4) -> str:
        """Improved LLM call with better prompts and fallback strategies"""
        
        # Build options text
        options_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
        
        # Build improved prompt
        if rag_context:
            prompt = f"""Context from knowledge base:
{rag_context}

Question: {question}

Options:
{options_text}

Based on the context above and your knowledge, select the best answer. Even if the context doesn't directly answer the question, use your reasoning to pick the most likely option.

Answer with just the letter (A, B, C, or D)."""
        else:
            prompt = f"""Question: {question}

Options:
{options_text}

Use your knowledge and reasoning to select the best answer.

Answer with just the letter (A, B, C, or D)."""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,  # Slightly increased for more thinking space
                "temperature": 0.1,
                "timeout": timeout
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            llm_time = time.time() - start_time
            print(f"🤖 LLM call: {llm_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                answer_text = result["choices"][0]["message"]["content"].strip()
                
                print(f"💭 LLM response: {answer_text}")
                
                # Parse answer letter - more flexible matching
                answer_match = re.search(r'\b([ABCD])\b', answer_text)
                if answer_match:
                    letter = answer_match.group(1)
                    letter_index = ord(letter) - ord('A')
                    if 0 <= letter_index < len(options):
                        selected_option = options[letter_index]
                        print(f"🎯 LLM selected: {letter} - '{selected_option}'")
                        return selected_option
                
                # If letter not found, try to find option text in response
                for option in options:
                    if option.lower() in answer_text.lower():
                        print(f"🎯 Found option in text: '{option}'")
                        return option
                
                # Final intelligent fallback: choose most likely answer
                print(f"⚠️ Could not parse LLM response, using intelligent fallback")
                return self.intelligent_fallback(question, options)
            
            else:
                print(f"❌ LLM API error: {response.status_code}")
                return self.intelligent_fallback(question, options)
                
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            return self.intelligent_fallback(question, options)
    
    def intelligent_fallback(self, question: str, options: List[str]) -> str:
        """Intelligent fallback strategy instead of blindly choosing first option"""
        
        question_lower = question.lower()
        
        # Strategy 1: Question keyword matching
        best_score = 0
        best_option = options[0]
        
        for option in options:
            option_lower = option.lower()
            # Calculate vocabulary overlap between question and option
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            option_words = set(re.findall(r'\b\w+\b', option_lower))
            overlap = len(question_words.intersection(option_words))
            
            if overlap > best_score:
                best_score = overlap
                best_option = option
        
        if best_score > 0:
            print(f"🧠 Intelligent fallback: '{best_option}' (keyword match)")
            return best_option
        
        # Strategy 2: Avoid obviously wrong answers
        avoid_patterns = ['never', 'impossible', 'all of the above', 'none of the above']
        for option in options:
            if not any(pattern in option.lower() for pattern in avoid_patterns):
                print(f"🎲 Safe fallback: '{option}'")
                return option
        
        # Final fallback
        print(f"📌 Default fallback: '{options[0]}'")
        return options[0]
    
    def run_quiz(self):
        """Run quiz loop"""
        print("🚀 Fixed Accuracy Kahoot Bot Starting...")
        print("🎯 This version focuses on accuracy improvements")
        
        try:
            with MCPServerAdapter(self.browser_server_params) as browser_tools:
                print("🔌 Browser MCP connected!")
                input("🔄 Connect extension and press Enter...")
                
                # Get browser tools
                snapshot_tool = None
                click_tool = None
                
                for tool in browser_tools:
                    if 'snapshot' in tool.name.lower():
                        snapshot_tool = tool
                    elif 'click' in tool.name.lower():
                        click_tool = tool
                
                if not snapshot_tool or not click_tool:
                    print("❌ Browser tools not found")
                    return
                
                print("✅ Browser tools ready!")
                
                question_count = 0
                correct_count = 0
                total_time = 0
                
                print("\n🎯 Starting accuracy-focused quiz mode!")
                
                while True:
                    try:
                        question_count += 1
                        start_time = time.time()
                        
                        print(f"\n{'='*60}")
                        print(f"🧠 Question #{question_count} (Accuracy-Focused)")
                        
                        # 1. Take page snapshot
                        print("📸 Taking snapshot...")
                        page_content = snapshot_tool._run()
                        
                        # 2. Improved question and options extraction
                        question_data = self.extract_question_and_options(str(page_content))
                        
                        if not question_data["valid"]:
                            print("❓ No valid or complete question found")
                            time.sleep(2)
                            continue
                        
                        question = question_data["question"]
                        options = question_data["options"]
                        refs = question_data["refs"]
                        
                        print(f"❓ Q: {question}")
                        print(f"📋 Options: {options}")
                        
                        # 3. Smart RAG query
                        rag_context = ""
                        if self.rag_tool:
                            print("🔍 Smart RAG query...")
                            rag_context = self.get_smart_rag_context(question)
                        
                        # 4. LLM call
                        print("🤖 Calling LLM with improved strategy...")
                        selected_answer = self.direct_llm_call(question, options, rag_context, timeout=5)
                        
                        # 5. Click answer
                        if selected_answer in options:
                            answer_index = options.index(selected_answer)
                            answer_ref = refs[answer_index] if answer_index < len(refs) else refs[0]
                            
                            print(f"🎯 Clicking: '{selected_answer}' (ref: {answer_ref})")
                            
                            try:
                                click_result = click_tool._run(element="button", ref=answer_ref)
                                print(f"✅ Click successful!")
                            except Exception as e:
                                print(f"❌ Click failed: {e}")
                        else:
                            print(f"❌ Selected answer not in options: {selected_answer}")
                        
                        elapsed = time.time() - start_time
                        total_time += elapsed
                        
                        # Performance statistics
                        avg_time = total_time / question_count
                        print(f"⏱️  This question: {elapsed:.2f}s")
                        print(f"📊 Average: {avg_time:.2f}s per question")
                        
                        # Accuracy feedback
                        is_correct = input("Was this answer correct? (y/n/skip): ").strip().lower()
                        if is_correct == 'y':
                            correct_count += 1
                        
                        if question_count > 0 and is_correct in ['y', 'n']:
                            accuracy = (correct_count / question_count) * 100
                            print(f"🎯 Current accuracy: {accuracy:.1f}% ({correct_count}/{question_count})")
                        
                        # Continue logic
                        if elapsed < 6:
                            print("⚡ Good speed! Continuing...")
                            time.sleep(1.5)
                        else:
                            next_action = input("🔄 Continue? (Enter/q): ").strip().lower()
                            if next_action == 'q':
                                break
                        
                    except KeyboardInterrupt:
                        print("\n🛑 Stopped by user")
                        break
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        continue_choice = input("🔄 Continue? (Enter/q): ").strip().lower()
                        if continue_choice == 'q':
                            break
                
                # Final statistics
                if question_count > 0:
                    avg_time = total_time / question_count
                    accuracy = (correct_count / question_count) * 100 if question_count > 0 else 0
                    
                    print(f"\n{'='*60}")
                    print("📊 FINAL STATISTICS")
                    print(f"{'='*60}")
                    print(f"Questions answered: {question_count}")
                    print(f"Correct answers: {correct_count}")
                    print(f"Accuracy: {accuracy:.1f}%")
                    print(f"Average time per question: {avg_time:.2f}s")
                    
                    if accuracy > 60:
                        print("🎉 Great accuracy!")
                    elif accuracy > 40:
                        print("👍 Good accuracy!")
                    else:
                        print("🔧 Need more tuning...")
                
        except Exception as e:
            print(f"❌ Session failed: {e}")

def main():
    bot = FixedAccuracyKahootBot()
    bot.run_quiz()

if __name__ == "__main__":
    main()