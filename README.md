<div align="center">
<h1>🧠 Memex<br/>Memory Infrastructure for AI Agents.</h1>
</div>


> Memory Infrastructure for AI Agents  
> Build persistent, structured, and evolving memory for intelligent systems.

Memex is an open-source memory infrastructure designed for AI agents and LLM-powered applications.

Modern AI systems are powerful but **stateless** — every interaction starts from zero.  
Memex provides a **persistent memory layer** that allows AI systems to remember users, experiences, knowledge, and context across time.

With Memex, AI applications can move from **stateless tools** to **learning systems that grow with experience**.

---


## ✨ Vision

AI today is powerful but forgetful.

Memex aims to build the **memory infrastructure for intelligent agents**, enabling them to:

- remember past interactions
- accumulate long-term knowledge
- understand persistent user context
- evolve through continuous learning

Our mission:

> Build the memory layer for the next generation of AI systems.

---

## 🚀 Key Features

### 🧠 Structured AI Memory

Memex transforms raw interactions into structured memory units.

Examples include:

- user preferences
- factual knowledge
- relationships
- events and experiences
- behavioral patterns

This allows AI systems to **store meaningful information instead of raw conversation logs**.

---

### 🔎 Hybrid Memory Retrieval

Memex supports multiple retrieval strategies:

- semantic vector retrieval
- keyword retrieval
- hybrid retrieval
- agentic multi-query recall

This enables both **fast recall and deep contextual reasoning**.

---

### 🗂 Hierarchical Memory Architecture

Memex organizes information into layered structures:

```
Experience → Memory Unit → Knowledge Structure
```

Example:

```
Conversation
      ↓
Extracted memories
      ↓
User profile / knowledge graph / episodic memory
```

Benefits include:

- traceable memory evolution
- structured reasoning
- efficient retrieval
- scalable knowledge growth

---

### 🔄 Self-Evolving Memory

Memex continuously improves stored memory through:

- summarization
- consolidation
- importance scoring
- temporal reasoning

Important memories become stronger over time, while irrelevant information gradually fades.

---

### 🎨 Multimodal Memory

Memex supports multiple data modalities.

| Modality | Examples |
|--------|--------|
| Conversation | chat history |
| Documents | notes, PDFs |
| Images | screenshots, photos |
| Audio | voice interactions |
| Video | recordings |

All modalities can be converted into unified memory representations.

---

## 🏗 Architecture

Memex is designed as a **modular memory infrastructure**.

```
                ┌───────────────────────┐
                │        AI Agent       │
                └─────────┬─────────────┘
                          │
                 Memory Query / Write
                          │
            ┌─────────────▼─────────────┐
            │         Memex Core        │
            │                           │
            │  Memory Extraction        │
            │  Memory Structuring       │
            │  Memory Consolidation     │
            │  Memory Retrieval         │
            └─────────────┬─────────────┘
                          │
             ┌────────────▼────────────┐
             │      Memory Storage      │
             │                          │
             │ Vector Database          │
             │ Graph Database           │
             │ Document Store           │
             └──────────────────────────┘
```

Core modules include:

- **Memory Extraction** — convert raw data into memory units  
- **Memory Structuring** — connect memories into structured knowledge  
- **Memory Retrieval** — retrieve relevant context for reasoning  
- **Memory Consolidation** — maintain long-term knowledge

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Yuan-ManX/Memex.git
cd Memex
pip install -e .
```

Or install via pip:

```bash
pip install memex
```

---

## ⚡ Quick Start

Initialize Memex:

```python
from memex import Memex

memex = Memex()
```

Store memory:

```python
memex.store(
    user_id="user_001",
    text="The user loves Chinese food."
)
```

Search memory:

```python
memories = memex.search(
    query="What food does the user like?"
)

print(memories)
```

---

## 🔌 API Example

### Store Memory

```
POST /api/memories
```

Example request:

```json
{
  "user_id": "user_001",
  "content": "The user prefers morning workouts."
}
```

---

### Retrieve Memory

```
GET /api/memories/search
```

Example request:

```json
{
  "query": "What habits does the user have?",
  "top_k": 5
}
```

---

## 🧠 Memory Types

Memex supports multiple memory categories.

| Type | Description |
|-----|-------------|
| Episodic Memory | Past interactions and experiences |
| Semantic Memory | Facts and knowledge |
| Profile Memory | User attributes |
| Preference Memory | User preferences |
| Relationship Memory | Social connections |

---

## 🎯 Use Cases

Memex can power many types of AI applications.

### AI Assistants
Personal assistants that remember users across conversations.

### AI Companions
AI companions capable of building emotional continuity.

### AI Agents
Autonomous agents capable of long-term planning and learning.

### AI Workflows
Systems that learn from previous tasks and improve performance.

---

## 🛣 Roadmap

Planned features include:

- memory graph engine
- multi-agent shared memory
- long-term knowledge compression
- temporal reasoning
- reinforcement learning from memory
- distributed memory infrastructure

---


## ⭐ Star History

If you find Memex useful, please consider giving the project a star ⭐

It helps the project grow and reach more developers.

---


## 📜 Contribution & License

Memex is **open source** and welcomes contributions from researchers, developers, and creators.

You can contribute by:

- Submitting new features or improvements
- Fixing bugs or optimizing performance
- Writing documentation, tutorials, or examples
- Reporting issues or suggesting enhancements

Please refer to [LICENSE](LICENSE).

---

---

## 🔮 The Future

The next generation of AI systems will not just generate responses.

They will **remember, learn, and evolve**.

Memex is building the **memory infrastructure for that future**.
