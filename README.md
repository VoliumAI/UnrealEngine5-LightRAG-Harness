# UnrealEngine5-LightRAG-Harness
# Unreal Engine 5 Documentation — LightRAG Retrieval Harness

A prebuilt retrieval corpus over Unreal Engine 5's official documentation: a FAISS vector index
+ a NetworkX knowledge graph + grounding images, built from `dev.epicgames.com`. This repo is
**retrieval only** — no fine-tuned model, no SFT data, nothing hosted. You connect it to
whatever model or agent you already use, and it hands back grounded UE5 context for that model
to answer from. Everything runs on CPU; the embedding model is ~146 MB.

## What's in this repo
<img width="1024" height="572" alt="b652b9d824e7498fa64d19d2596ddce6" src="https://github.com/user-attachments/assets/3e54fdf2-f455-4fcb-bece-2ee97f502bab" />



| File | What it is |
|---|---|
| `ue5_faiss_v4.index` | FAISS vector index of embedded documentation chunks |
| `ue5_chunks_v4.json` | The text/metadata for each chunk, in the same order as the FAISS index |
| `ue5_knowledge_graph_v4.graphml` | NetworkX graph connecting UE5 classes, subsystems, and functions |

The embedding model and the reference images are hosted separately (see Setup below) — this
keeps the GitHub repo small, since neither fits comfortably in a git repo.

Keep everything in the same folder once you've downloaded it all — chunk records reference the
images folder by relative path, and the FAISS index only lines up with `ue5_chunks_v4.json` if
both are loaded together, as saved.

## Setup (do this once, regardless of what you connect it to)

**1. Get the embedding model.** Every vector in `ue5_faiss_v4.index` was built with **Snowflake
Arctic Embed M Long, Q8_0, GGUF**. You must embed queries with this exact model — a different
model puts queries in a different vector space and retrieval will return nonsense.

Download: **https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF**
(direct file, 146 MB: [`.../resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf`](https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF/resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf))

Put it in a `models/` folder next to your code.

**2. Get the reference images.** Diagrams/screenshots from the source docs ship as
`UE5_images.rar` on the same Hugging Face repo as the embedding model:
**https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF**

Download it and extract it into the same folder as `ue5_faiss_v4.index`/`ue5_chunks_v4.json` —
extracting it should produce a `ue5_images_v4/` subfolder there (any archive tool works: 7-Zip,
WinRAR, `unrar x UE5_images.rar`, etc.). If you don't need image grounding, you can skip this —
everything else works without it, you'll just get an empty image list from that part of the API.

**3. Install dependencies.**

```bash
pip install faiss-cpu llama-cpp-python numpy networkx
```

**4. Save the retrieval helper.** This repo intentionally doesn't ship a scraper or any serving
code — just the three data files above. Save this as `ue5_retrieve.py` in the same folder:

```python
import json
import numpy as np
import faiss
import networkx as nx
from llama_cpp import Llama

MODEL_PATH       = "models/snowflake-arctic-embed-m-long-q8_0.gguf"   # point at your download
FAISS_INDEX_FILE = "ue5_faiss_v4.index"
CHUNK_JSON_FILE  = "ue5_chunks_v4.json"
GRAPH_FILE       = "ue5_knowledge_graph_v4.graphml"

_model = Llama(model_path=MODEL_PATH, embedding=True, n_ctx=1024, n_gpu_layers=0, verbose=False)
_index = faiss.read_index(FAISS_INDEX_FILE)
with open(CHUNK_JSON_FILE, "r", encoding="utf-8") as f:
    _chunks = json.load(f)
_graph = nx.read_graphml(GRAPH_FILE)

# Map each graph node's human-readable label -> its node id. We don't need to
# know how those ids were generated, just that every node carries a "name"
# (entities) or "title" (pages) attribute holding the real text.
_name_to_node = {}
for _nid, _data in _graph.nodes(data=True):
    _label = _data.get("name") or _data.get("title")
    if _label:
        _name_to_node.setdefault(_label, _nid)


def embed(text: str) -> np.ndarray:
    vec = np.array(_model.embed(text), dtype=np.float32)
    if vec.ndim > 1:
        vec = vec.mean(axis=0)          # some pooling configs return one row per token
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def graph_context(entities: list, hops: int = 2) -> list:
    lines, frontier, seen = [], [], set()
    for e in entities:
        nid = _name_to_node.get(e)
        if nid:
            frontier.append(nid)
            seen.add(nid)
    for _ in range(hops):
        nxt = []
        for nid in frontier:
            src_label = _graph.nodes[nid].get("name") or _graph.nodes[nid].get("title")
            for _, dst, edata in _graph.out_edges(nid, data=True):
                dst_label = _graph.nodes[dst].get("name") or _graph.nodes[dst].get("title")
                lines.append(f"{src_label} -[{edata.get('relation', 'related_to')}]-> {dst_label}")
                if dst not in seen:
                    seen.add(dst)
                    nxt.append(dst)
        frontier = nxt
    return lines


def retrieve(query: str, top_k: int = 5, graph_hops: int = 2, with_images: bool = False):
    q = embed(query).reshape(1, -1)
    scores, idxs = _index.search(q, top_k)
    hits = [_chunks[i] for i in idxs[0] if 0 <= i < len(_chunks)]

    parts = [f"[{i + 1}] ({h['url']}) {h['text']}" for i, h in enumerate(hits)]
    all_entities = {e for h in hits for e in h.get("entities", [])}
    lines = graph_context(list(all_entities), hops=graph_hops)
    if lines:
        parts.append("## Connected Concepts:\n" + "\n".join(lines))
    ctx = "\n\n".join(parts)

    if not with_images:
        return ctx

    image_paths = []
    for h in hits:
        for img in h.get("images", []):
            if img["path"] not in image_paths:
                image_paths.append(img["path"])
    return ctx, image_paths
```

**5. Confirm it works.**

```python
from ue5_retrieve import retrieve
print(retrieve("How do I enable Lumen via Python?", top_k=3))
```

⚠️ **If step 1 wasn't done correctly, this raises an error from `llama_cpp` (bad model path)
rather than silently returning nothing** — that's the first thing to check if this doesn't work.

That's the whole retrieval API. Everything below is just different ways to put that string in
front of a model.

---

## Connect it to your setup

Pick one:

- Already using **Claude Desktop, Claude Code, or Cursor** → **Option A (MCP)**
- Calling a **hosted model's API directly** (Anthropic, OpenAI, or similar) → **Option B**
- Running a **local model or your own agent loop** → **Option C**

### Option A — MCP (Claude Desktop, Claude Code, Cursor, or any MCP client)

Set this up once and the model can call it whenever it decides UE5 context would help, instead
of you pasting context in by hand every time.

```bash
pip install fastmcp
```

Save this as `ue5_mcp_server.py` in the same folder as `ue5_retrieve.py`:

```python
from fastmcp import FastMCP
from ue5_retrieve import retrieve

mcp = FastMCP("ue5-lightrag")

@mcp.tool()
def search_ue5_docs(query: str, top_k: int = 5) -> str:
    """Search Unreal Engine 5 documentation and return grounded context
    (relevant chunks + connected knowledge-graph concepts) for the query."""
    return retrieve(query, top_k=top_k, graph_hops=2)

@mcp.tool()
def search_ue5_docs_with_images(query: str, top_k: int = 5) -> dict:
    """Same as search_ue5_docs, plus local file paths of any diagrams/
    screenshots attached to the retrieved chunks."""
    ctx, image_paths = retrieve(query, top_k=top_k, with_images=True)
    return {"context": ctx, "image_paths": image_paths}

if __name__ == "__main__":
    mcp.run()   # stdio transport — the default, and what Claude Desktop/Code expect
```

Register it. For **Claude Code**:

```bash
claude mcp add ue5-lightrag python /absolute/path/to/ue5_mcp_server.py
```

For **Claude Desktop** (or Cursor, same shape), add this to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`) and restart the app:

```json
{
  "mcpServers": {
    "ue5-lightrag": {
      "command": "python",
      "args": ["/absolute/path/to/ue5_mcp_server.py"]
    }
  }
}
```

Use **absolute paths** everywhere — for the script, and for `MODEL_PATH` inside
`ue5_retrieve.py` — MCP launches your script as a subprocess from wherever the client lives, not
from this folder, so relative paths silently fail to resolve.

### Option B — A cloud API (Anthropic, OpenAI, or any chat-completions endpoint)

Retrieve first, then put the result in the system prompt (or prepend it to the user message for
endpoints with no separate system field) before sending the request — standard RAG wiring.

**Anthropic:**

```python
from ue5_retrieve import retrieve
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
user_query = "How do I enable Lumen via Python?"
ctx = retrieve(user_query, top_k=5, graph_hops=2)

response = client.messages.create(
    model="claude-sonnet-5",
    system=f"You are a UE5 automation assistant. Use this retrieved context when relevant:\n\n{ctx}",
    messages=[{"role": "user", "content": user_query}],
    max_tokens=1024,
)
print(response.content[0].text)
```

**OpenAI (or any OpenAI-compatible endpoint):**

```python
from ue5_retrieve import retrieve
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from the environment
user_query = "How do I enable Lumen via Python?"
ctx = retrieve(user_query, top_k=5, graph_hops=2)

response = client.chat.completions.create(
    model="<your model>",   # whichever model you currently have access to
    messages=[
        {"role": "system", "content": f"You are a UE5 automation assistant. Use this retrieved context when relevant:\n\n{ctx}"},
        {"role": "user", "content": user_query},
    ],
)
print(response.choices[0].message.content)
```

Any other provider follows one of these two shapes (separate `system` field, or a
`role: "system"` entry inside `messages`) — check which your SDK uses and drop `ctx` in there.

### Option C — A local model or your own agent framework

If you're running a local llama.cpp server, Ollama, or a custom agent loop, the integration is
the same one line — `retrieve()` returns plain text (or text + image paths), so it doesn't care
what consumes it:

```python
from ue5_retrieve import retrieve

ctx, image_paths = retrieve(user_query, top_k=5, with_images=True)
# ctx          -> paste into your prompt template wherever "context" or "retrieved docs" goes
# image_paths  -> local file paths; load and attach these if your model accepts image input
```

For a vision-capable model, load each path in `image_paths` and attach it as an image content
block alongside `ctx`.

---

## Reference

**Chunk fields** (`ue5_chunks_v4.json`): `text` (embedded content), `title`, `url` (source page
on `dev.epicgames.com`), `entities` (UE5 class/subsystem names detected in that chunk),
`sections` (the source page's own subheadings), `images` (list of `{"path", "alt"}`, path
relative to `ue5_images_v4/`).

**Knowledge graph**: a `networkx.DiGraph` in `ue5_knowledge_graph_v4.graphml`. Entity/relation
extraction is regex-based pattern matching over the doc text, not a real parser of Unreal's
class hierarchy — useful as a "what else is this connected to" hint, not ground truth.

**This is a static snapshot.** It reflects Epic's documentation as of whenever this corpus was
built and won't pick up anything published after that. The tooling that generated it isn't part
of this repo.

**Source content**: text and images are derived from Epic Games' own Unreal Engine
documentation — short excerpts and a modest number of diagrams, not a full mirror. Check Epic's
documentation terms yourself before redistributing the corpus further.

**Source content**: text and images are derived from Epic Games' own Unreal Engine
documentation — short excerpts and a modest number of diagrams, not a full mirror. Check Epic's
documentation terms yourself before redistributing the corpus further.
