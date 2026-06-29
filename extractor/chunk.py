from dataclasses import dataclass, field

@dataclass
class Chunk: 
    source:str # title of the research paper
    chunk_id:str
    chunk_type:str
    parent:int #id of the parent chunk if no parents then -1
    text:str
    metadata : dict = field(default_factory=dict)
 
    def to_dict(self) -> dict:
        return {
            "id"       : self.chunk_id,
            "body"     : self.text,
            "metadata" : self.metadata,
        }
    
    def __str__(self):
        return self.text
    
    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(id={self.chunk_id!r}, source: {self.source}, section: {self.metadata["section_header"]}, body: {preview!r}..., type: {self.chunk_type})"