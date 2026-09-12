# UnrealEngine5-LightRAG-Harness
# Unreal Engine 5 Documentation — LightRAG Retrieval Harness

A prebuilt retrieval corpus over Unreal Engine 5's official documentation: a FAISS vector index
+ a NetworkX knowledge graph + grounding images, built from `dev.epicgames.com`. This repo is
**retrieval only** — no fine-tuned model, no SFT data, nothing hosted. You connect it to
whatever model or agent you already use, and it hands back grounded UE5 context for that model
to answer from. Everything runs on CPU; the embedding model is ~146 MB.

## What's in this repo

| File | What it is |
|---|---|
| `ue5_faiss_v4.index` | FAISS vector index of embedded documentation chunks |
| `ue5_chunks_v4.json` | The text/metadata for each chunk, in the same order as the FAISS index |
| `ue5_knowledge_graph_v4.graphml` | NetworkX graph connecting UE5 classes, subsystems, and functions |
| `ue5_images_v4/` | Diagrams/screenshots from the source docs, referenced by the chunks that use them |
| `UnrealDocu.py` | The scraper that built the four files above, plus the retrieval functions used below |

Keep all of the above in the same folder — chunk records reference the images folder by
relative path, and the FAISS index only lines up with `ue5_chunks_v4.json` if both are loaded
from where they were saved together.

## Setup (do this once, regardless of what you connect it to)

**1. Get the embedding model.** Every vector in `ue5_faiss_v4.index` was built with **Snowflake
Arctic Embed M Long, Q8_0, GGUF**. You must embed queries with this exact model — a different
model puts queries in a different vector space and retrieval will return nonsense.

Download: **https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF**
(direct file, 146 MB: [`.../resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf`](https://huggingface.co/Volium/snowflake-arctic-embed-m-long-q8_0.GGUF/resolve/main/snowflake-arctic-embed-m-long-q8_0.gguf))

Put it in a `models/` folder next to your code, or point straight at it:

```bash
# Windows (PowerShell)
$env:APP_MODELS_DIR = "C:\path\to\snowflake-arctic-embed-m-long-q8_0.gguf"
# macOS / Linux
export APP_MODELS_DIR=/path/to/snowflake-arctic-embed-m-long-q8_0.gguf
```

**2. Install dependencies.**

```bash
pip install faiss-cpu llama-cpp-python numpy networkx requests beautifulsoup4
```

(`requests`/`beautifulsoup4` are only used by the scraper half of `UnrealDocu.py`, but Python
imports the whole file top-to-bottom the moment you import anything from it, so they're required
even if you only ever call the two functions below.)

**3. Confirm it works.**

```python
from UnrealDocu import build_lightrag_context
print(build_lightrag_context("How do I enable Lumen via Python?", top_k=3))
```

⚠️ **If step 1 wasn't done correctly, this prints an empty string instead of erroring** — no
embedding model found means retrieval silently returns nothing rather than crashing. If you get
`""` back, that's the first thing to check, not a bug in the code below.

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

Save this as `ue5_mcp_server.py` in the same folder as the data files:

```python
from fastmcp import FastMCP
from UnrealDocu import build_lightrag_context, build_lightrag_context_with_images

mcp = FastMCP("ue5-lightrag")

@mcp.tool()
def search_ue5_docs(query: str, top_k: int = 5) -> str:
    """Search Unreal Engine 5 documentation and return grounded context
    (relevant chunks + connected knowledge-graph concepts) for the query."""
    return build_lightrag_context(query, top_k=top_k, graph_hops=2)

@mcp.tool()
def search_ue5_docs_with_images(query: str, top_k: int = 5) -> dict:
    """Same as search_ue5_docs, plus local file paths of any diagrams/
    screenshots attached to the retrieved chunks."""
    ctx, image_paths = build_lightrag_context_with_images(query, top_k=top_k)
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

Use **absolute paths** for both the script and the model file (`APP_MODELS_DIR`) — MCP launches
your script as a subprocess from wherever the client lives, not from this folder, so relative
paths silently fail to resolve (see step 3's warning above — this is the most common way to hit
that empty-string result).

### Option B — A cloud API (Anthropic, OpenAI, or any chat-completions endpoint)

Retrieve first, then put the result in the system prompt (or prepend it to the user message for
endpoints with no separate system field) before sending the request — standard RAG wiring.

**Anthropic:**

```python
from UnrealDocu import build_lightrag_context
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
user_query = "How do I enable Lumen via Python?"
ctx = build_lightrag_context(user_query, top_k=5, graph_hops=2)

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
from UnrealDocu import build_lightrag_context
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from the environment
user_query = "How do I enable Lumen via Python?"
ctx = build_lightrag_context(user_query, top_k=5, graph_hops=2)

response = client.chat.completions.create(
    model="<your model>",   # e.g. whichever GPT model you currently have access to
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
the same one line — `build_lightrag_context()` returns plain text, so it doesn't care what
consumes it:

```python
from UnrealDocu import build_lightrag_context_with_images

ctx, image_paths = build_lightrag_context_with_images(user_query, top_k=5)
# ctx          -> paste into your prompt template wherever "context" or "retrieved docs" goes
# image_paths  -> local file paths; load and attach these if your model accepts image input
```

For a vision-capable model, load each path in `image_paths` and attach it as an image content
block alongside `ctx` — that's what `build_lightrag_context_with_images()` is for, versus the
text-only `build_lightrag_context()`.

---

## Reference

**Chunk fields** (`ue5_chunks_v4.json`): `text` (embedded content), `title`, `url` (source page
on `dev.epicgames.com`), `entities` (UE5 class/subsystem names detected in that chunk),
`sections` (the source page's own subheadings), `images` (list of `{"path", "alt"}`, path
relative to `ue5_images_v4/`).

**Knowledge graph**: a `networkx.DiGraph` in `ue5_knowledge_graph_v4.graphml`. Entity/relation
extraction is regex-based pattern matching over the doc text, not a real parser of Unreal's
class hierarchy — useful as a "what else is this connected to" hint, not ground truth.

**Regenerating the corpus**: `python UnrealDocu.py` re-scrapes and rebuilds all four output
files (it checkpoints, so an interrupted run resumes). Not needed for normal use — only if you
want to refresh after Epic updates their docs, or extend coverage.

**Source content**: text and images are derived from Epic Games' own Unreal Engine
documentation — short excerpts and a modest number of diagrams, not a full mirror. Check Epic's
documentation terms yourself before redistributing the corpus further.
