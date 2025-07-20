#!/usr/bin/env python3
"""
Alternative document processing solution
Fallback solution when Docling fails on M1 chips
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import subprocess

class AlternativeDocumentProcessor:
    """Alternative document processor - multiple methods"""
    
    def __init__(self):
        self.available_methods = self._check_available_methods()
        print(f"🔍 Available parsing methods: {list(self.available_methods.keys())}")
    
    def _check_available_methods(self) -> Dict[str, bool]:
        """Check which parsing methods are available"""
        methods = {}
        
        # Check Unstructured
        try:
            import unstructured
            methods['unstructured'] = True
            print("✅ Unstructured available")
        except ImportError:
            methods['unstructured'] = False
            print("❌ Unstructured not available (pip install unstructured[pdf])")
        
        # Check PyMuPDF
        try:
            import pymupdf
            methods['pymupdf'] = True
            print("✅ PyMuPDF available")
        except ImportError:
            methods['pymupdf'] = False
            print("❌ PyMuPDF not available (pip install pymupdf)")
        
        # Check pdfplumber
        try:
            import pdfplumber
            methods['pdfplumber'] = True
            print("✅ pdfplumber available")
        except ImportError:
            methods['pdfplumber'] = False
            print("❌ pdfplumber not available (pip install pdfplumber)")
        
        # Check system tools
        methods['pdftotext'] = self._check_pdftotext()
        methods['textract'] = self._check_textract()
        
        return methods
    
    def _check_pdftotext(self) -> bool:
        """Check pdftotext command line tool"""
        try:
            result = subprocess.run(['pdftotext', '-v'], 
                                  capture_output=True, text=True)
            print("✅ pdftotext command line tool available")
            return True
        except:
            print("❌ pdftotext not available (brew install poppler or apt-get install poppler-utils)")
            return False
    
    def _check_textract(self) -> bool:
        """Check textract"""
        try:
            import textract
            print("✅ textract available")
            return True
        except ImportError:
            print("❌ textract not available (pip install textract)")
            return False
    
    def extract_text_best_method(self, file_path: str) -> Dict[str, Any]:
        """Extract text using the best available method"""
        
        # Try different methods by priority
        methods_priority = [
            ('unstructured', self._extract_with_unstructured),
            ('pdfplumber', self._extract_with_pdfplumber), 
            ('pymupdf', self._extract_with_pymupdf),
            ('pdftotext', self._extract_with_pdftotext),
            ('textract', self._extract_with_textract)
        ]
        
        for method_name, extract_func in methods_priority:
            if self.available_methods.get(method_name, False):
                print(f"🔄 Trying to extract document with {method_name}...")
                try:
                    result = extract_func(file_path)
                    if result and result.get('content') and len(result['content'].strip()) > 100:
                        print(f"✅ {method_name} extraction successful!")
                        result['extraction_method'] = method_name
                        return result
                    else:
                        print(f"⚠️ {method_name} extracted too little content")
                except Exception as e:
                    print(f"❌ {method_name} extraction failed: {e}")
                    continue
        
        print("💥 All methods failed")
        return None
    
    def _extract_with_unstructured(self, file_path: str) -> Dict[str, Any]:
        """Extract using Unstructured"""
        from unstructured.partition.pdf import partition_pdf
        
        # Unstructured advantage: specialized for complex document layouts
        elements = partition_pdf(
            filename=file_path,
            strategy="fast",  # Fast mode, avoid complex AI processing
            infer_table_structure=False,  # Skip table recognition
            extract_images_in_pdf=False,  # Skip image extraction
        )
        
        # Extract text content
        content_parts = []
        for element in elements:
            if hasattr(element, 'text'):
                content_parts.append(element.text)
        
        content = '\n\n'.join(content_parts)
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'title': Path(file_path).name,
                'elements_count': len(elements),
                'extraction_method': 'unstructured'
            }
        }
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber"""
        import pdfplumber
        
        content_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                except Exception as e:
                    print(f"⚠️ Page {page_num + 1} extraction failed: {e}")
                    continue
        
        content = '\n\n'.join(content_parts)
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'title': Path(file_path).name,
                'page_count': len(pdf.pages),
                'extraction_method': 'pdfplumber'
            }
        }
    
    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract using PyMuPDF"""
        import pymupdf as fitz
        
        doc = fitz.open(file_path)
        content_parts = []
        
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            except Exception as e:
                print(f"⚠️ Page {page_num + 1} extraction failed: {e}")
                continue
        
        page_count = len(doc)
        doc.close()
        
        content = '\n\n'.join(content_parts)
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'title': Path(file_path).name,
                'page_count': page_count,
                'extraction_method': 'pymupdf'
            }
        }
    
    def _extract_with_pdftotext(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdftotext command line tool"""
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_file:
            try:
                # Run pdftotext command
                result = subprocess.run([
                    'pdftotext', 
                    '-layout',  # Preserve layout
                    '-enc', 'UTF-8',  # Use UTF-8 encoding
                    file_path, 
                    tmp_file.name
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    with open(tmp_file.name, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    os.unlink(tmp_file.name)  # Delete temporary file
                    
                    return {
                        'content': content,
                        'metadata': {
                            'source': file_path,
                            'title': Path(file_path).name,
                            'extraction_method': 'pdftotext'
                        }
                    }
                else:
                    raise Exception(f"pdftotext failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                raise Exception("pdftotext timeout")
            except Exception as e:
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                raise e
    
    def _extract_with_textract(self, file_path: str) -> Dict[str, Any]:
        """Extract using textract"""
        import textract
        
        # textract automatically detects file type
        content = textract.process(file_path).decode('utf-8')
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'title': Path(file_path).name,
                'extraction_method': 'textract'
            }
        }

def install_alternatives():
    """Install alternative solutions"""
    print("🛠️ Installing alternative document processing tools")
    
    packages = [
        "pip install unstructured[pdf]",
        "pip install pymupdf", 
        "pip install pdfplumber",
        "pip install textract",
    ]
    
    print("Please run the following commands to install alternative tools:")
    for pkg in packages:
        print(f"  {pkg}")
    
    print("\nSystem tools (optional):")
    print("  macOS: brew install poppler")
    print("  Ubuntu: sudo apt-get install poppler-utils")

def test_extraction(file_path: str):
    """Test document extraction"""
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return
    
    processor = AlternativeDocumentProcessor()
    result = processor.extract_text_best_method(file_path)
    
    if result:
        content = result['content']
        method = result.get('extraction_method', 'unknown')
        
        print(f"\n🎉 Extraction successful!")
        print(f"📊 Method used: {method}")
        print(f"📝 Content length: {len(content)} characters")
        print(f"📋 Content preview:\n{content[:500]}...")
        
        # Save to file for viewing
        output_file = f"extracted_{Path(file_path).stem}_{method}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Full content saved to: {output_file}")
        
    else:
        print("💥 All extraction methods failed")
        print("💡 Suggestions:")
        print("   1. Check if PDF file is corrupted")
        print("   2. Try converting PDF format with other tools")
        print("   3. If it's a scanned document, you might need OCR tools")

def main():
    """Main function"""
    print("🔧 Alternative Document Processor Test")
    print("="*40)
    
    # Check available methods
    processor = AlternativeDocumentProcessor()
    
    if not any(processor.available_methods.values()):
        print("❌ No available processing methods")
        install_alternatives()
        return
    
    # Test file
    file_path = input("Please enter PDF file path: ").strip().strip('"\'')
    test_extraction(file_path)

if __name__ == "__main__":
    main()