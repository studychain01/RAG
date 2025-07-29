# Graph RAG: Knowledge Graph Enhanced Retrieval-Augmented Generation

> **Combines the power of knowledge graphs with vector databases to dramatically improve question-answering performance, especially for multi-hop queries.**

## 🎯 What You Will Learn

This project demonstrates how to overcome the limitations of traditional RAG systems by integrating knowledge graphs with vector databases, enabling superior performance on complex, multi-hop queries.

## 🚫 Limitations of Traditional RAG

Traditional RAG systems face several critical challenges when dealing with complex queries:

### 1. **Multi-hop Questions** 🔄

**Problem:** Traditional RAG struggles with questions that require reasoning across multiple facts or steps.

**Example:**
> *"What contribution did the son of Euler's teacher make?"*

**To answer this requires:**
1. **Step 1:** Identify Euler's teacher
2. **Step 2:** Identify that teacher's son  
3. **Step 3:** Determine that son's contribution

**Why it's hard for RAG:**
Traditional RAG retrieves documents based on similarity to the entire query. If no single passage covers all three facts together, the model can't easily stitch the steps across different documents.

### 2. **Complex Entity Relationships** 🔗

**Problem:** It's difficult for RAG to understand nuanced relationships between entities.

**Example:**
> *"Which physicists were influenced by Heisenberg's uncertainty principle?"*

**This question assumes:**
- Knowledge of Heisenberg's principle
- Awareness of who was exposed to or referenced it
- A link between those individuals and Heisenberg

**Why it's hard for RAG:**
RAG may retrieve texts about Heisenberg and others separately but not the link showing influence or citation. Entity relationships often span documents or are implicit.

### 3. **Context Fragmentation** 📄

**Problem:** Relevant information is often scattered across different passages that aren't co-located in the corpus.

**Example:**
- One document mentions *"Einstein attended the Solvay Conference."*
- Another says *"Bohr and Einstein debated quantum theory at the conference."*

Neither document alone gives you the complete context of Einstein's debate with Bohr, even though both are relevant.

**Why it's hard for RAG:**
Traditional RAG retrieves and summarizes in isolation. Without stitching or merging related passages, the full picture gets lost.

### 4. **Semantic Gaps** 🕳️

**Problem:** RAG relies on semantic similarity (via embeddings). But sometimes logically important information doesn't "sound" similar.

**Example:**
- **Query:** *"What led to the collapse of the housing market in 2008?"*
- **Relevant fact:** *"Excessive leveraging of subprime mortgages created systemic risk"*

These two pieces may not be semantically close in vector space, so the model may not retrieve the most important evidence.

**Why it's hard for RAG:**
If the wording is different—even if the logic aligns—embedding-based search might miss it.

---

## ✨ Key Benefits

| Feature | Description |
|---------|-------------|
| **🔧 Simplified Architecture** | Single vector database instead of vector DB + graph DB combination |
| **🚀 Superior Multi-hop Performance** | Handles complex queries requiring multiple relationship traversals |
| **📈 Scalable** | Leverages Milvus's distributed architecture for billion-scale deployments |
| **💰 Cost-effective** | Reduces infrastructure complexity and operational overhead |
| **🎛️ Flexible** | Works with any text corpus - just extract entities and relationships |

---

## 🛠️ Modern Solutions

Let me know if you want visual examples, code analogies, or how modern techniques like **Graph RAG**, **ToT RAG**, or **HyDE** tackle these issues.

---

## 🔬 Methodology Overview

Our approach consists of four main stages that work together to overcome traditional RAG limitations:

### 1. **Offline Data Preparation** 📚

- Extract entities and relationships (triplets) from your text corpus
- Create three vector collections: entities, relationships, and passages
- Build adjacency mappings between entities and relationships

### 2. **Query-time Retrieval** 🔍

- Retrieve similar entities and relationships using vector similarity search
- Use Named Entity Recognition (NER) to identify query entities

### 3. **Subgraph Expansion** 🌐

- Expand retrieved entities/relationships to their neighborhood using adjacency matrices
- Support multi-degree expansion (1-hop, 2-hop neighbors)
- Merge results from both entity and relationship expansion paths

### 4. **LLM Reranking** 🧠

- Use large language models to intelligently filter and rank candidate relationships
- Apply Chain-of-Thought reasoning to select most relevant relationships
- Return final passages for answer generation

---

## 🏗️ Architecture Diagram

The following diagram illustrates our complete Graph RAG workflow:

![Graph RAG Architecture](graph_rag_architecture.png)

### **System Overview:**

**🔄 Offline Loading Stage:**
- **Corpus Processing:** Raw text data is ingested and processed
- **Entity & Relationship Extraction:** Knowledge graph components are extracted and structured
- **Vector Database Storage:** All entities and relationships are embedded and stored in Milvus Vector DB

**⚡ Online Retrieval Stage:**
- **Query Processing:** User queries are processed and embedded for vector search
- **Parallel Retrieval:** Both entities and relationships are retrieved simultaneously from the vector database
- **LLM Reranking:** Retrieved candidates are intelligently filtered and ranked by the LLM
- **Answer Generation:** Final answer is generated using the refined knowledge graph information

### **Key Advantages:**
- **Dual Retrieval:** Simultaneous entity and relationship retrieval ensures comprehensive coverage
- **Intelligent Filtering:** LLM reranking eliminates irrelevant candidates
- **Scalable Architecture:** Milvus handles billion-scale vector operations efficiently

---

## 🛠️ Technology Stack

### **Named Entity Recognition (NER)**
Used for entity extraction from text:
- **spaCy** - Industrial-strength NLP library
- **Hugging Face Transformers** - State-of-the-art transformer models
- **Stanza** (by Stanford) - Multi-language NLP toolkit
- **Flair** - Simple framework for state-of-the-art NLP
- **OpenAI/GPT-4/Claude** - Large language model APIs

### **Relationship Extraction**
Tools for extracting entity relationships:
- **OpenIE** (Stanford) - Open Information Extraction
- **spaCy + Dependency Parsing** - Rule-based relationship extraction
- **LLMs** - Large language models for relationship inference
- **REBEL** - Relation extraction with BERT
- **KAIROS** - Temporal and causal relationship extraction
- **DyGIE++** - Dynamic Graph Information Extraction

---

*Ready to explore the future of RAG? Dive into the code and see how knowledge graphs can transform your question-answering capabilities!*





maybe most importatnt piece of information 


Understanding the Data Model

Before diving into the implementation, it's crucial to understand how we structure our data to enable graph-like reasoning with vectors. Our approach transforms traditional text documents into three interconnected components:

    Entities: The "nodes" of our conceptual graph - people, places, concepts, etc.
    Relationships: The "edges" connecting entities - these are full triplets (subject-predicate-object)
    Passages: The original text documents that provide context and detailed information

Why This Structure Works: By separating entities and relationships into distinct vector collections, we can perform targeted searches for different aspects of a query. When a user asks "What contribution did the son of Euler's teacher make?", we can:

    Find entities related to "Euler"
    Find relationships that connect teacher-student and parent-child concepts
    Expand the graph to discover indirect connections
    Retrieve the most relevant passages for final answer generation


Data Preparation

We will use a nano dataset which introduce the relationship between Bernoulli family and Euler to demonstrate as an example. The nano dataset contains 4 passages and a set of corresponding triplets, where each triplet contains a subject, a predicate, and an object.

Triplet Structure: Each relationship is represented as a triplet [Subject, Predicate, Object]. For example:

    ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"] captures a family relationship
    ["Johann Bernoulli", "was a student of", "Leonhard Euler"] captures an educational relationship

In practice, you can use any approach to extract the triplets from your own custom corpus. Common methods include:

    Named Entity Recognition (NER) + Relation Extraction models
    Open Information Extraction systems like OpenIE
    Large Language Models with structured prompting
    Manual annotation for high-precision domains




We construct the entities and relations as follows:

    The entity is the subject or object in the triplet, so we directly extract them from the triplets.
    Here we construct the concept of relationship by directly concatenating the subject, predicate, and object with a space in between.

We also prepare a dict to map entity id to relation id, and another dict to map relation id to passage id for later use.


Building the Knowledge Graph Structure

The next step transforms our triplets into a searchable vector format while maintaining the graph connectivity information. This process involves several key decisions:

Entity Extraction Strategy: We extract unique entities by collecting all subjects and objects from our triplets. This ensures we capture every entity mentioned in any relationship, creating comprehensive coverage of our knowledge domain.

Relationship Representation: Rather than storing relationships as separate subject-predicate-object components, we concatenate them into natural language sentences. For example, ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"] becomes "Jakob Bernoulli was the older brother of Johann Bernoulli". This approach offers several advantages:

    Semantic richness: The full sentence provides more context for vector embeddings
    Natural language compatibility: LLMs can easily understand and reason about complete sentences
    Reduced complexity: No need to manage separate predicate vocabularies


Adjacency Mapping Construction: We build two critical mapping structures:

    entityid_2_relationids: Maps each entity to all relationships it participates in (enables entity-to-relationship expansion)
    relationid_2_passageids: Maps each relationship to the passages where it appears (enables relationship-to-passage retrieval)

These mappings are essential for the subgraph expansion process, allowing us to efficiently traverse the conceptual graph during query time.


Data Insertion

Create Milvus collections for entity, relation, and passage. We create three separate Milvus collections, each optimized for different types of retrieval:

    Entity Collection: Stores vector embeddings of entity names and descriptions
        Purpose: Enables entity-centric queries like "find entities similar to 'Euler'"
        Search pattern: Direct semantic similarity to query entities

    Relationship Collection: Stores vector embeddings of complete relationship sentences
        Purpose: Captures semantic patterns in relationships that match query intent
        Search pattern: Finds relationships semantically similar to the entire query

    Passage Collection: Stores vector embeddings of original text passages
        Purpose: Provides comparison baseline and detailed context for final answers
        Search pattern: Traditional RAG-style document retrieval

Why Three Collections? This separation allows for multi-modal retrieval:

    If a query mentions specific entities, we retrieve through the entity collection
    If a query describes relationships or actions, we retrieve through the relationship collection
    We can combine results from both paths and compare against traditional passage retrieval

Embedding Consistency: All collections use the same embedding model to ensure compatibility during similarity searches and result merging.


