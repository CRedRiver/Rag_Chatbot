import os
import re
from typing import List
import pymupdf4llm
from extractor.text_splitter import RecursiveCharacterTextSplitter
from extractor.chunk import Chunk


class PdfExtractor:
    """
    ## Extract Research papers into chunks. ##
    Chunks Hierarchy: Parent Chunk -> Child chunks 
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._current_id = 0
        
        self.section_keywords = [
            "abstract", "introduction", "background", "literature review", 
            "method", "methods", "methodology", "results", "findings", 
            "discussion", "conclusion", "conclusions", "acknowledgements", 
            "acknowledgments", "references"
        ]
        
        kw_pattern = '|'.join(self.section_keywords)
        self.header_regex = re.compile(rf'^(?:\d+(?:\.\d+)*\.?\s*|[ivx]+\.?\s*|[a-z]\.\s*)?({kw_pattern})$', re.IGNORECASE)

    def pdf_to_md(self, file_path: str) -> str:
        return pymupdf4llm.to_markdown(file_path)
        
    def _get_next_id(self) -> int:
        self._current_id += 1
        return self._current_id

    def extract_parent_chunk(self, parent_chunk: Chunk) -> List[Chunk]:
        child_chunks = []
        table_pattern = re.compile(r'((?:\|.*\|\r?\n)+)')
        blocks = table_pattern.split(parent_chunk.text)
        
        for block in blocks:
            if not block.strip():
                continue
            
            # Handle Markdown Tables
            if block.strip().startswith('|') and '|' in block:
                rows = [row.strip() for row in block.strip().split('\n') if row.strip()]
                header = rows[0] if len(rows) > 0 else ""
                
                for i, row in enumerate(rows):
                    # Check for structural separators, including colons for alignment (e.g., |:---:|)
                    if set(row.replace('|', '').replace('-', '').replace(':', '').replace(' ', '')) == set():
                        continue
                    
                    row_text = f"Table Header: {header}\nRow Data: {row}" if i > 1 else row
                    # Inherit parent metadata
                    child_meta = {"is_table": True, "row_index": i}
                    child_meta.update(parent_chunk.metadata)
                    
                    child_chunk = Chunk(
                        source=parent_chunk.source,
                        chunk_id=self._get_next_id(),
                        chunk_type="Table Row",
                        parent=parent_chunk.chunk_id,
                        text=row_text,
                        metadata=child_meta
                    )
                    child_chunks.append(child_chunk)
                    
            # Handle Regular Text
            else:
                splits = self.text_splitter.split_text(block)
                for split in splits:
                    # Inherit parent metadata
                    child_meta = {"is_table": False}
                    child_meta.update(parent_chunk.metadata)
                    
                    child_chunk = Chunk(
                        source=parent_chunk.source,
                        chunk_id=self._get_next_id(),
                        chunk_type="Paragraph",
                        parent=parent_chunk.chunk_id,
                        text=split,
                        metadata=child_meta
                    )
                    child_chunks.append(child_chunk)
                    
        return child_chunks

    def extract(self, file_path: str):
        """
        ## Extract pdf file ##
        Returns: tuple(list[Chunk], list[Chunk])\n
        * A list of all parent chunks
        * A list of all children chunks

        """
        self._current_id = 0 
        md_text = self.pdf_to_md(file_path)
        
        # Always return a tuple of (parent_chunks, children_chunks)
        if not md_text:
            return [], []

        # Split into lines but KEEP blank lines so \n\n splitting works
        lines = md_text.splitlines()

        # Extract Title 
        paper_title = "Unknown Title"
        skip_words = ["research paper", "article", "review article", "original article"] 
        
        for line in lines:
            clean_line = line.replace('#', '').replace('*', '').strip()
            
            # Skip empty lines and generic tags
            if not clean_line or clean_line.lower() in skip_words:
                continue
                
            # Grab the first line that starts with an alphabetical character
            if clean_line[0].isalpha():
                paper_title = clean_line
                break

        # Group text into sections based on Keywords
        sections = []
        current_header = "General/Metadata"
        current_text_blocks = []
        
        for line in lines:
            clean_line = line.replace('#', '').replace('*', '').strip()
            is_header = False
            
            # Check if the line is a section header 
            if clean_line:
                if self.header_regex.match(clean_line):
                    # Save the previous section
                    if current_text_blocks:
                        # Join with newlines to preserve formatting
                        sections.append((current_header, "\n".join(current_text_blocks)))
                    
                    # Start new section
                    current_header = clean_line 
                    current_text_blocks = []
                    is_header = True
            
            # If it's not a header, append it to the current section's text
            if not is_header:
                current_text_blocks.append(line)
                
        # Append the final section
        if current_text_blocks:
            sections.append((current_header, "\n".join(current_text_blocks)))
            
        # Generate Chunks for each section
        parent_chunks = []
        children_chunks = []
        for header, text in sections:
            combined_text = f"{header}\n{text}".strip()
            if not combined_text:
                continue
                
            parent_chunk = Chunk(
                source=paper_title,
                chunk_id=self._get_next_id(),
                chunk_type="Section",
                parent=-1,
                text=combined_text,
                metadata={"section_header": header}
            )
            parent_chunks.append(parent_chunk)
            
            children = self.extract_parent_chunk(parent_chunk)
            children_chunks.extend(children)
            
        return parent_chunks, children_chunks

    def extract_batch(self, file_path: List[str]):
        par, chil = [], []
        for path in file_path:
            par_, chil_ = self.extract(path)
            par.extend(par_)
            chil.extend(chil_)
        return par, chil


if __name__ == "__main__":
    path = r"D:\HUST\2025.2\Project1_new\data\EJ1172284.pdf"
    try:
        extractor = PdfExtractor()
        text = extractor.pdf_to_md(path)
        print(f"----EXTRACTING FILE------")
        print(f"Preview: {text[:100]}")
        try: 
            par, chil = extractor.extract_batch([path])
            for chunk in par:
                print(repr(chunk))
        except Exception as e:
            print(f"Error chunking the file:", e)

    except Exception as e:
        print(f"Error extracting text: {e}")
    