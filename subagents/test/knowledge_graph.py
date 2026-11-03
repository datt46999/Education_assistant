
from subagents.Graph_RAG.knowledge_graph import KnowleadgeGraph
from dotenv import load_dotenv
load_dotenv()
"""
================================================
Test 
================================================
"""
SAMPLE_DOCUMENTS = {
    "research_team.txt": """
Dr. Sarah Chen leads the Advanced AI Research Lab at TechCorp. 
She pioneered the development of the NeuraSparse algorithm in 2022.
Dr. Chen previously studied under Professor James Miller at Stanford University.
Professor Miller is known for his foundational work in neural network optimization.
The Advanced AI Research Lab collaborates closely with the Quantum Computing Division.
Dr. Chen's team includes 15 researchers working on efficient neural architectures.
""",

    "quantum_project.txt": """
The Quantum Computing Division at TechCorp is led by Dr. Marcus Williams.
Dr. Williams implemented the NeuraSparse algorithm in their quantum error correction system.
This implementation reduced error rates by 47% compared to traditional methods.
The quantum team works on Project Aurora, funded by a $50 million grant.
Project Aurora aims to build a fault-tolerant quantum computer by 2027.
The project uses insights from Professor Miller's optimization research.
""",

    "company_overview.txt": """
TechCorp was founded in 2015 by Elena Rodriguez in San Francisco.
The company has grown to over 500 employees across three divisions.
TechCorp's three main divisions are: AI Research, Quantum Computing, and Cloud Services.
Elena Rodriguez previously worked at Google as a principal engineer.
The company's headquarters are located in the South Bay area.
TechCorp has partnerships with MIT and Stanford University for research collaboration.
"""
}
if __name__ == "__main__":
    kg_rag = KnowleadgeGraph()


    # Ingest all documents
    for doc_name, content in SAMPLE_DOCUMENTS.items():
        kg_rag.add_document(content, doc_name)

    ADDITIONAL_QUERIES = [
        "What is the relationship between Dr. Chen and Project Aurora?",
        "How did the NeuraSparse algorithm impact quantum error correction?",
        "Who founded the company where Dr. Chen works?"
    ]

    for query in ADDITIONAL_QUERIES:
        print("\n" + "="*70)
        print(f"❓ {query}")
        print("="*70)
        
        context = kg_rag.hybrid_search(query, vector_top_k=3, graph_hops=2)
        result = kg_rag.generate_attributed_answer(query, context)
        
        print(f"\n💬 Answer:\n{result.answer[:500]}...")