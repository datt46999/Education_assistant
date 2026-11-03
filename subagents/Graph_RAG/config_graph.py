import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field

MODEL_NAME = "gpt-4o-mini"
# EMBEDDING_MODEL= "text-embedding-3-small"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


"""
==========================================================================================
Data model for Graph RAG define structure classes to track: Entities, relationship citation
==========================================================================================
"""

@dataclass
class Entity:
    """Entities extracted from a document and full provenmance"""
    id: str
    name: str
    entity_type:str #person, organization, concept, technology
    description: str
    source_document: str
    source_sentence: str
    embedding: Optional[np.ndarray] = None

@dataclass
class Relationship:
    """A relationship between two entities with provenance"""
    source_entity: str
    target_entity: str
    relation_type: str  # works_for, location_in, develop
    description: str
    source_document: str
    source_sentence: str

@dataclass
class Citation: 
    """A verifiable citation linking a claim to its more"""
    claim: str
    source_document: str
    source_sentence: str
    confidence: str
    graph_path: List[str] #path through the graph that led to the claim


@dataclass
class AttributeAnswer:
    """Final answer with full attribute"""
    answer: str
    citations: List[Citation]
    reasoning_trace: List[str]
    entities_used: List[str]
    relationships_traverved: List[str]
