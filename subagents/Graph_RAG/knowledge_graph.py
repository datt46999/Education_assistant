import os
import networkx as nx
import json
import hashlib
import re 
import numpy as np
import re


from sentence_transformers import SentenceTransformer

from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from subagents.Graph_RAG.config_graph import *

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embedder = SentenceTransformer(EMBEDDING_MODEL)
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

_gemini_clients: Dict[str, Any] = {}


def _call_gemini(prompt: str, model: str, json_mode: bool = False)->str:
    key = f"{model}:{json_mode}"

    if key not in _gemini_clients:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        extra = {"response_mime_type": "application/json"} if json_mode else {}
        _gemini_clients[key] = ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=0.0, max_retries=3, **extra
        )
    content = _gemini_clients[key].invoke(prompt).content
    if isinstance(content, list):  # Gemini can return a list of content parts
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content or ""


def _strip_json_fences(text: str)-> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text


def call_llm(prompt: str, model: str= MODEL_NAME) ->str:
    """
    Call local LLM via Ollama
    """
    # if not OLLAMA_AVAILABLE:
    #     return '{"entities": [], "relationships": []}'

    # response = ollama.chat(
    #     model = model,
    #     messages = [{"role": "user", "content": prompt}]
    # )
    
    if model.startswith("gemini"):
        return _call_gemini(prompt, model)

    if MODEL_NAME == "gpt-4o-mini":
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""
    # return response["message"]["content"]
    


def call_llm_json(prompt, model: str= MODEL_NAME)-> str:
    """call llm and parse json response"""
    try:
        # if OLLAMA_AVAILABLE:
        #     response = ollama.chat(
        #         model = model,
        #         messages = [{"role":"user", "content": prompt}],
        #         format = "json"
        #     )
        #     return json.load(response["message"]["content"])


        if model.startswith("gemini"):
            return json.loads(_strip_json_fences(_call_gemini(prompt, model, json_mode=True)))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)

    except Exception as error:
        print(f"call_llm_json failed: {error}")
        return {"entities": [], "relationship": [], "error": str(error)}


"""
===============================================
Knowledge graph manager
+ extract entities and relationship using llm
+ build graph structured
+ Maintain source provenance for every node and edge
===============================================
"""

class KnowledgeGraph:
    """
    Knowledge graph with verifiable attribute

    Class demonstrate graph Rag:
    + run entities local 
    + extract entities and relationship using llm
    + enable multi-hop relationship
    + provides verifiable source attribution
    """

    def __init__(self, model_name = MODEL_NAME):
        """Initialize knowledge graph"""
        self.graph = nx.DiGraph()
        self.model_name = model_name
        self.entities : Dict[str, Entity] = {}
        self.relationships : List[Relationship] = []
        self.documents: Dict[str, str] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}


    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for an entity."""

        return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]

    def extract_entities_and_relationship(self, text: str, doc_name: str) -> tuple[list[Entity], list[Relationship]]:
        """
        Use llm to extract entities and relationship from document
        Each entities preserve source sentence for attribute
        """
        extract_prompt = """Analyze this text and extract entities and relationships.
        TEXT:
        {text}
        Return Json with this extract structure:
        {{
            "entities":[
                {{
                    "name": "Entity name",
                    "type": "PERSON|ORGANIZATION|CONCEPT|TECHNOLOGY|EVENT|LOCATION",
                    "description": "Brief description",
                    "source_sentence": "Extract sentence from the text where this entities appears" 
                }}
            ],
            "relationship":[
                {{
                    "source": "Source Entities Name",
                    "target": "Target Entities Name",
                    "type": "WORKS_FOR|DEVELOPED|USES|LOCATED_IN|RELATED_TO|FOUNDED|MENTORED",
                    "description": "How they are related",
                    "source_sentence": "Extract sentence  descripting this relationship"
                }}
            ]
        }}

        IMPORTANT:
        + Including the EXACT source_sentence for orginal text.
        + This is critical for verifiable attribution.
        """

        results = call_llm_json(extract_prompt.format(text = text), model = self.model_name)
        entities = []
        for e in results.get("entities", []):
            if not isinstance(e, dict) or not e.get("name"):
                continue
            entity_id = self._generate_id(e['name'])
            entity = Entity(
                id=  entity_id,
                name = e['name'],
                entity_type= e.get('type', "UNKNOWN"),
                description= e.get("description", ""),
                source_document= doc_name,
                source_sentence= e.get("source_sentence", text[:100])
            )
            entities.append(entity)


        relationship = []
        for r in results.get("relationships", []) or results.get("relationships", []):
            if not isinstance(r, dict) or not r.get("source") or not r.get("target"):
                continue
            rel = Relationship(
                source_entity= r["source"],
                target_entity= r["target"],
                relation_type= r.get('type', 'UNKNOWN'),
                description= r.get("description", ""),
                source_document= doc_name,
                source_sentence= r.get("source_sentence", "")
            )

            relationship.append(rel)

        return entities, relationship   

    def add_document(self, text: str, doc_name: str)-> tuple[list[Entity], list[Relationship]]:
        """
        ingest document into knowledge app.
        extrac entities and relationship, preversing source string
        """
        
        self.documents[doc_name] = text
        print(f"Procesing: {doc_name}")

        entities, relationships = self.extract_entities_and_relationship(text, doc_name)


        # add entities to graph:
        for entity in entities:
            self.entities[entity.id] = entity


            # generate embedding for vector search
            embedding = embedder.encode(f"{entity.name}: {entity.description}")
            self.entity_embeddings[entity.id] = embedding
            entity.embedding = embedding

            # add node to graph with all prevernance metadata
            self.graph.add_node(
                entity.id,
                name = entity.name,
                type = entity.entity_type,
                description = entity.description,
                source_document = entity.source_document,
                source_sentence = entity.source_sentence
            )

        # add relationship to  graph
        for rela in relationships:
            source_id = self._generate_id(rela.source_entity)
            target_id = self._generate_id(rela.target_entity)

            if source_id in self.entities and target_id in self.entities:
                self.graph.add_edge(
                    source_id, 
                    target_id,
                    relation_type = rela.relation_type,
                    description = rela.description,
                    source_document = rela.source_document,
                    source_sentence = rela.source_sentence
                ) 
                self.relationships.append(rela)

        print(f"Extracted {len(entities)} entities and {len(relationships)}  relationships")
        return entities, relationships

    def vector_search(self, query: str, top_k: int = 3):
        """
        find the most entities to the query from vector search
        """

        query_embedding = embedder.encode(query)
        similarities = []
        for entity_id, embedding in self.entity_embeddings.items():
            similarity = np.dot(query_embedding, embedding)/(
                np.linalg.norm(query_embedding)*np.linalg.norm(embedding)
            )
            similarities.append((entity_id, similarity))
        # sort by similarity:
        similarities.sort(key= lambda x: x[1] , reverse= True)
        return [self.entities[eid] for eid, _ in similarities[:top_k]]


    def multi_hop_traversed(self, start_entities: List[Entity], max_hops: int= 2)-> List[Dict[str, Any]]:
        """
        Perform mutil-hop traversed graph from start entitiy
        Return earch reachable node with their pairs (for attribution)
        """

        results = []
        visited = set()

        for start_entity in start_entities:
            # BFS traversal up to max_hop

            queue = [(start_entity.id, [start_entity.name], 0)]

            while queue:
                current_id, path, depth = queue.pop(0)
                if depth > max_hops or current_id in visited:
                    continue 


                visited.add(current_id)

                if current_id in self.graph:
                    node_data = self.graph.nodes[current_id]
                    results.append({
                        "entity_id": current_id,
                        "entity_name": node_data.get("name"),
                        "description": node_data.get("description"),
                        "source_document": node_data.get("source_document"),
                        "source_sentence": node_data.get("source_sentence"),
                        "path":path,
                        "depth":depth
                    })

                    # for neighbor in self.graph.neighbors(current_id):
                    #     edge_data = self.graph.edges[current_id, neighbor]
                    #     neighbor_name = self.graph.nodes[neighbor].get['name', neighbor]
                    #     new_path = path + [f"--[{edge_data.get('relation_type')}]-->", neighbor_name]
                    #     queue.append(neighbor, new_path, depth+1)

                    for neighbor in self.graph.neighbors(current_id):
                        edge_data = self.graph.edges[current_id, neighbor]
                        neighbor_name = self.graph.nodes[neighbor].get('name', neighbor)
                        new_path = path + [f"--[{edge_data.get('relation_type')}]-->", neighbor_name]
                        queue.append((neighbor, new_path, depth + 1))

        return results


    def hybrid_search(self, query: str, vector_top_k: int = 3,
                       graph_hops: int = 2):
        """
        Hybried search combining vectore similarty and graph traverser

        1. vector search find entity points(most relevent entities point)
        2. Graph traversed expand the context informaion
        """

        entry_point = self.vector_search(query, top_k = vector_top_k)

        print(f"Found {len(entry_point)} entry point for vector search")

        traversed_results = self.multi_hop_traversed(entry_point, graph_hops)
        return traversed_results


    def generate_attributed_answer(self, query: str, context: List[Dict[str, Any]])->AttributeAnswer:
        """
        Generate the answer with verifiable source  citation

        Each claim answer will link to:
        + source document
        + source sentence
        + Graph path led to this information

        Args:
            query (str): query string to processing
            context (list[dict[str, any]]): retrieve use to generate answer

        Return:
            AttributeAnswer: (structure container)
                answers: the generate answer
                citations: citation support answer
                reasoning_trace: the reasoning/retrieval step use
                entities_use: the Entities use for the knowledge graph and use for construct the answer
                relationship_use: Graph relationship traversed
        """
        context_paths = []
        source_map = {}

        for i, item in enumerate(context):
            source_key = f"{i+1}"
            

            context_paths.append(f"[{source_key}] {item['entity_name']}: {item['description']}")

            source_map[source_key] = {
                "document":item["source_document"], 
                "sentence":item["source_sentence"],
                "path": item["path"]
            }

        context_text = "\n".join(context_paths)


        answer_prompt = f"""Base on this knowledge context. Answer the question.
        CRITICAL: Cite your source using [N] for each claim.
        CONTEXT:

        {context_text}

        QUESTION: {query}

        Provide a comprehensive answer with inline critical like [1], [2] for each claim
        """



        # answer_text = call_llm(answer_prompt)
        # citations_ref = re.findall(r"\[(\d+)\]", answer_text)

        answer_text = call_llm(answer_prompt, model=self.model_name)
        citations_ref = re.findall(r"\[(\d+)\]", answer_text)


        citations = []
        for ref in sorted(set(citations_ref), key=int):
            if ref in source_map:                 # ref = "1", khớp key của source_map
                src = source_map[ref]
                citations.append(Citation(
                    claim=f"Reference [{ref}]",
                    source_document=src["document"],
                    source_sentence=src["sentence"],
                    confidence=0.9,
                    graph_path=src["path"],
                ))


        reasoning_trace = [
            f"🔍 Query: {query}",
            f"📌 Found {len(context)} relevant entities",
            f"🔗 Traversed graph paths",
            f"📝 Generated answer with {len(citations)} citations"
        ]

        return AttributeAnswer(
            answer= answer_text,
            citations=citations,
            reasoning_trace=reasoning_trace,
            entities_used=[c['entity_name'] for c in context],
            relationships_traverved=[]
        )


