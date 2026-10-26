# TransformerScope — Complete Project Guide

Deep-dive documentation of every file, function, and data flow in this project.
Read this to understand how the app works end-to-end without reading source first.

---

## 1. What This Project Is

Interactive **Streamlit web app** that loads a pre-trained transformer model, runs your text through it, extracts raw attention weights from every layer/head, and renders them as interactive Plotly visualizations.

**Stack:** Python · Streamlit · Hugging Face Transformers · PyTorch · Plotly · Pandas · NumPy

**Purpose:** Education, model interpretability, and debugging — see which tokens "attend" to which other tokens inside a transformer.

---

## 2. Project Structure

```
TransformerScope/
├── app.py                          # Main Streamlit app — UI + orchestration (277 lines)
├── src/
│   ├── __init__.py                 # Empty package marker
│   ├── attention_extractor.py      # Model loading, attention extraction, stats (176 lines)
│   └── visualizer.py               # All Plotly chart builders (318 lines)
├── examples/
│   └── sample_texts.txt            # Extra sample texts for manual testing
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview + quick start
├── PROJECT_GUIDE.md                # This file — detailed internals
└── LICENSE                         # MIT license
```

**Separation of concerns:**
- `attention_extractor.py` → data layer (model in, numpy arrays out)
- `visualizer.py` → presentation layer (numpy arrays in, Plotly figures out)
- `app.py` → glue layer (UI widgets, user flow, calls into both)

---

## 3. Data Flow (End-to-End)

```
User text
   │
   ▼
① load_model_and_tokenizer(model_name)          [attention_extractor.py]
   │  HF AutoTokenizer + AutoModel(output_attentions=True), cached via @st.cache_resource
   ▼
② extract_attention_weights(model, tokenizer, text)
   │  tokenize → forward pass under torch.no_grad()
   │  returns: attentions (tuple of tensors), tokens (list[str]), token_ids
   ▼
③ get_attention_stats(attentions) / analyze_head_patterns(attentions, tokens)
   │  computes layer entropies, token importance, per-head metrics
   ▼
④ process_attention_data(attentions, tokens, layer, head)
   │  slices out one (seq_len × seq_len) matrix + cleans token labels
   ▼
⑤ visualizer.create_*(...)  →  Plotly Figure
   ▼
⑥ st.plotly_chart(fig)  →  rendered in browser
```

### Attention tensor shapes (key concept)

| Stage | Shape |
|---|---|
| `attentions` | tuple of length `num_layers` |
| each element | `(batch=1, num_heads, seq_len, seq_len)` |
| after `process_attention_data` | `(seq_len, seq_len)` numpy array |

Row `i` = attention distribution **from** token `i` (query) **to** all tokens (keys).
Each row sums to ~1.0 (softmax over keys).

---

## 4. File Deep-Dive

### 4.1 `app.py` — Main Application (277 lines)

Streamlit script that runs top-to-bottom on every widget interaction.

**Page setup (lines 19–30):**
- `page_title="TransformerScope"`, wide layout, sidebar expanded.

**Sidebar — Configuration (lines 33–48):**
- Model dropdown with 4 choices (default `distilbert-base-uncased`):
  - `distilbert-base-uncased` (6 layers, 12 heads — fast)
  - `bert-base-uncased` (12 layers, 12 heads)
  - `distilgpt2` (6 layers, 12 heads — decoder-only)
  - `gpt2` (12 layers, 12 heads — decoder-only)

**Main area — two columns:**
- **Left col:** text input. Radio picks a sample (`Simple`, `Complex`, `Question`, `Attention Example`, `Custom`) → `st.text_area` for the actual text.
- **Right col:** analysis options:
  - `analysis_type` selectbox: `Attention Heatmap` | `Token Importance` | `Layer Comparison` | `Head Patterns` | `Attention Flow`
  - `normalize_attention` checkbox (default True)
  - `show_stats` checkbox (default True)

**Analyze button flow (lines 96–268):** guarded by `if st.button(...)` and non-empty text.

1. Load model (spinner shown while downloading/running).
2. Extract attentions.
3. Compute stats.
4. Show 4 metrics: model name, token count, layer count, heads/layer.
5. Render **tokenized input** as HTML chips (strips GPT-2 `Ġ` and BERT `##` prefixes; blank tokens shown as `▁`).
6. If analysis type needs it (`Heatmap`, `Head Patterns`, `Flow`): show **layer** selectbox (default layer 2) and **head** selectbox (Heatmap/Flow only).
7. Dispatch to visualization branch:

| Analysis Type | Functions Called | Output |
|---|---|---|
| Attention Heatmap | `process_attention_data` → `create_attention_heatmap` + `create_token_attention_bar` | Heatmap + bar chart |
| Token Importance | inline: sorts `stats['total_attention']` | Top-5 metric cards |
| Layer Comparison | `create_layer_comparison` ×2 (`entropy`, `max_attention`) | 2 line charts |
| Head Patterns | `analyze_head_patterns` → `create_head_pattern_comparison` + metric cards | Multi-line chart + 4 head metrics |
| Attention Flow | `process_attention_data` → `create_attention_flow_diagram` | Circular network graph |

8. If `show_stats`: layer-by-layer entropy list + architecture info (layers, heads, seq length, ~param count in millions).

**Error handling (lines 261–266):** any exception → `st.error` message, hint to shorten text, collapsible `traceback` expander.

---

### 4.2 `src/attention_extractor.py` — Data Layer (176 lines)

Imports: `torch`, `numpy`, `transformers.AutoTokenizer/AutoModel`, `streamlit`.

#### `load_model_and_tokenizer(model_name)` — line 9
- Decorated `@st.cache_resource` → model loaded **once** per process, reused across reruns.
- `AutoModel.from_pretrained(..., output_attentions=True)` — critical flag; without it `outputs.attentions` is `None`.
- `model.eval()` disables dropout.
- Error mapping: 404/repo-not-found → "check model name"; connection/timeout → network hint; memory/cuda → suggest `distilbert-base-uncased`; fallback generic message. Returns `(None, None)` on failure.

#### `extract_attention_weights(model, tokenizer, text, max_length=512)` — line 32
- Tokenizes with `return_tensors="pt"`, `truncation=True`, `padding=True`.
- `tokens = tokenizer.convert_ids_to_tokens(...)` for display.
- Forward pass inside `torch.no_grad()` (no gradient memory).
- Returns:
  - `attentions` — tuple of tensors, one per layer
  - `tokens` — list of token strings
  - `token_ids` — input ID tensor

#### `process_attention_data(attentions, tokens, layer_idx=0, head_idx=0)` — line 63
- Validates layer/head index (raises `ValueError` with available counts).
- Slices `attentions[layer_idx][0, head_idx, :, :]` → numpy `(seq_len, seq_len)`.
- Cleans labels: strips BERT `##`, wraps special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`) in brackets.
- Returns `(attention_matrix, token_labels)`.

#### `get_attention_stats(attentions)` — line 101
Returns dict:
- `num_layers`, `num_heads`, `sequence_length`
- `layer_entropies` — per-layer mean entropy: average attention over batch+heads, then `H = -Σ a·log(a+1e-8)` per query row, mean over rows. **Low entropy = focused attention; high = spread out.**
- `total_attention` — `(seq_len,)` array: sums each layer's attention over dims `(batch, heads, source positions)` → measures how much attention each token **receives** overall. Feeds Token Importance view.

#### `analyze_head_patterns(attentions, tokens)` — line 139
Nested dict: `patterns['layer_{i}']['head_{j}'] = {...}` with:
- `self_attention` — mean of diagonal (how much a token attends to itself)
- `entropy` — mean normalized row entropy (spread/focus)
- `max_attention` — single strongest weight
- `attention_matrix` — raw `(seq_len, seq_len)` numpy array

Note: `tokens` param accepted but unused (metrics are position-based).

---

### 4.3 `src/visualizer.py` — Presentation Layer (318 lines)

Imports: `plotly.graph_objects`, `plotly.express`, `numpy`, `pandas`, `streamlit`.
All functions take plain data and return a Plotly `Figure` — no Streamlit calls, fully testable.

#### `create_attention_heatmap(attention_matrix, token_labels, layer_idx, head_idx, normalize=True)` — line 10
- Optional row-normalization (each row sums to 1).
- `go.Heatmap` with `Blues` colorscale; hover shows `From / To / Attention`.
- X = "Attended To (Keys)", Y = "Attending From (Queries)", x labels rotated 45°.
- 700×600.

#### `create_token_attention_bar(attention_matrix, token_labels, normalize=True)` — line 42
- Column-sums the matrix → total attention **received** per token.
- `px.bar`, color-mapped by value (`Blues`), 800×400.

#### `create_layer_comparison(attentions, tokens, metric='entropy')` — line 80
- Loops all layers × heads.
  - `entropy`: per-head mean row entropy → layer mean ± std.
  - `max_attention`: per-head max value → layer mean ± std.
- `px.line` with `error_y` = std band, 700×400.

#### `create_head_pattern_comparison(layer_patterns, layer_idx)` — line 155
- Takes one layer's dict from `analyze_head_patterns`.
- Three overlaid `go.Scatter` lines: Self Attention (blue), normalized Entropy (red), Max Attention (green).
- Entropy divided by its max to bring all metrics onto comparable 0–1 scale.

#### `create_attention_flow_diagram(attention_matrix, token_labels, threshold=0.1)` — line 219
- Collects edges where weight > threshold **and** `i != j` (self-loops excluded).
- Fallback: if nothing passes, re-threshold at `max(weight) * 0.5`.
- Nodes placed on a unit circle (`angle = 2πi/n`).
- Two `go.Scatter` traces: edges (lightblue lines) + nodes (lightcoral markers with text labels).
- Hides axes/grid, white background, 600×600.

---

## 5. The 5 Visualization Types — What They Mean

| # | View | Question It Answers |
|---|---|---|
| 1 | **Attention Heatmap** | For a chosen layer+head: which token pair has strongest connection? |
| 2 | **Token Importance** | Which tokens received the most attention across the whole model? |
| 3 | **Layer Comparison** | How does attention focus/entropy change with model depth? |
| 4 | **Head Patterns** | Within one layer, which heads are self-attentive vs. spread out? |
| 5 | **Attention Flow** | What are the dominant non-self attention links, as a network graph? |

**Interpreting results:**
- High **self-attention** → token mostly looks at itself (common in later layers).
- **Sequential attention** → attends to previous token (positional/syntactic).
- **Semantic attention** → jumps to related words (e.g., "She" → "Mary").
- **Low entropy layer** → decisive, focused attention. **High entropy** → diffuse.
- Encoder models (BERT/DistilBERT) show bidirectional patterns; decoder models (GPT-2) show causal lower-triangular patterns (each token only sees earlier tokens).

---

## 6. Supported Models

| Model ID | Type | Layers | Heads | Character |
|---|---|---|---|---|
| `distilbert-base-uncased` | Encoder | 6 | 12 | Lightweight, fastest, good default |
| `bert-base-uncased` | Encoder | 12 | 12 | Classic bidirectional baseline |
| `distilgpt2` | Decoder | 6 | 12 | Small causal LM |
| `gpt2` | Decoder | 12 | 12 | Generative, 124M params |

First run downloads weights from Hugging Face Hub (needs internet). Cached afterward via `@st.cache_resource`.

---

## 7. Setup & Run

```bash
# 1. Clone
git clone https://github.com/Yashrajsalunkhe/TransformerScope.git
cd TransformerScope

# 2. Virtual env (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Run
streamlit run app.py
# → opens http://localhost:8501
```

### Dependencies (`requirements.txt`)

| Package | Purpose | Used? |
|---|---|---|
| `streamlit>=1.28.0` | Web UI | ✅ |
| `torch>=2.0.0` | Tensor ops / model inference | ✅ |
| `transformers>=4.35.0` | Model + tokenizer loading | ✅ |
| `numpy>=1.24.0` | Array math | ✅ |
| `plotly>=5.17.0` | Interactive charts | ✅ |
| `pandas>=2.0.0` | DataFrames for charts | ✅ |
| `scikit-learn>=1.3.0` | — | ❌ unused |
| `matplotlib>=3.7.0` | — | ❌ unused |
| `seaborn>=0.12.0` | — | ❌ unused |

Last three could be removed from requirements (no import anywhere).

---

## 8. Key Implementation Details

- **Caching:** only model/tokenizer cached (`st.cache_resource`). Tokenization + forward pass re-run on every button click — correct, since text can change.
- **Button-gated analysis:** all heavy work inside `if st.button(...)`; results disappear on next widget interaction (Streamlit rerun model). Not stored in `st.session_state`.
- **`output_attentions=True`** must be passed at model load — set once in the cached loader.
- **Normalization** applied at visualization time (checkbox), not extraction — raw weights preserved.
- **`1e-8` epsilon** used throughout to avoid `log(0)` / division by zero.
- **Token cleaning** done in two places: `process_attention_data` (labels) and `app.py` inline (HTML chips).
- **No batching:** single text, batch size 1, max 512 tokens with truncation.
- **CPU by default** — code never calls `.cuda()` / `.to(device)`; runs anywhere.

---

## 9. Known Limitations / Improvement Ideas

| Area | Current State | Possible Improvement |
|---|---|---|
| Session state | Results vanish on widget change | Store results in `st.session_state` |
| Device | CPU only | Auto-detect CUDA/MPS |
| Models | 4 hardcoded | Custom model name input, decoder+encoder auto-detect |
| Export | None | PNG/CSV download buttons |
| Attention view | Raw weights only | Rollout / attention-flow aggregation across layers |
| Flow diagram | Edge width fixed at 1 | Scale line width by attention weight |
| Batch input | Single text | Process multiple texts |
| Causal masking | Not visualized | Gray out impossible (future) positions for GPT-2 |
| Unused deps | 3 packages | Prune `requirements.txt` |
| Tests | None | Unit-test extractor + visualizer with tiny model |

---

## 10. Quick Reference — Where to Change What

| I want to... | Edit |
|---|---|
| Add a new model to the dropdown | `app.py` line ~36 `model_options` |
| Add a sample text | `app.py` line ~57 `sample_texts` dict |
| Add a new analysis type | `app.py`: add to `analysis_type` list + new `elif` branch; add chart fn in `visualizer.py` |
| Change entropy formula | `attention_extractor.py` → `get_attention_stats` / `analyze_head_patterns` |
| Change colors/layout of a chart | `visualizer.py` → respective `create_*` function |
| Adjust flow-graph threshold | `visualizer.py` → `create_attention_flow_diagram(threshold=0.1)` |
| Support longer inputs | `app.py` → `extract_attention_weights(..., max_length=512)` call site + default |
| Change default layer/head | `app.py` line ~151 (`index=min(2, ...)`) and head `index=0` |

---

*Companion to `README.md` (quick start) — this file covers internals. MIT License.*
