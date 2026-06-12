# KarimGPT — Web App

A ChatGPT-style web UI for all three from-scratch KarimGPT models, with a
model picker to switch between them.

| Model | Params | Type | Weights |
|---|---|---|---|
| KarimGPT 1 | 13M | Character-level (Shakespeare) | `../V1 (13m Param)/model.pt` |
| KarimGPT 2 | 33M | BPE chat (50k conversations) | `../V2 (33m Param)/model_bpe.pt` |
| KarimGPT 3 | 91M | GPT-2 style (largest) | `../V3 (91m Param)/model_gpt2.pt` |

```
KarimGPT/
├── V1 (13m Param)/        # training code + weights (never modified)
├── V2 (33m Param)/
├── V3 (91m Param)/
└── webapp/
    ├── server.py          # FastAPI backend — serves all 3 models
    ├── requirements.txt
    ├── vercel.json        # static frontend deploy config (public/ only)
    └── public/            # the ChatGPT-style frontend (deploy this to Vercel)
        ├── index.html
        ├── styles.css
        └── app.js
```

The original training scripts (`gpt.py`, `KarimGPT_bpe.py`, `gpt2_train.py`)
and the weights are **never modified or copied** — `server.py` re-declares the
architectures and loads the weights by reference from the version folders.

Models are **lazy-loaded**: the server starts instantly and only loads a model
into RAM the first time someone selects it (the UI shows "Loading KarimGPT N…"
during a cold start). To preload instead: `KARIMGPT_PRELOAD=v2,v3`.

---

## 1. Run the backend (serves the models)

```powershell
cd webapp
pip install -r requirements.txt
python server.py
```

This starts the API on **http://localhost:8000** and also serves the frontend
there, so you can just open <http://localhost:8000> and start chatting locally.

Endpoints:
- `GET  /api/health` → server status + which models are loaded
- `GET  /api/models` → model catalog for the picker
- `POST /api/chat`   → streams the reply token-by-token (SSE); body includes `model: "v1" | "v2" | "v3"`

Env vars:
- `KARIMGPT_PRELOAD=v1,v2,v3` — eager-load models at startup
- `KARIMGPT_V1_DIR` / `V2` / `V3` — override weight folder locations (for deployment)
- `KARIMGPT_ALLOW_UNSAFE_LOAD=1` — only needed if a checkpoint was saved as a
  full pickled module rather than a state dict (off by default for safety)

---

## 2. Deploy

**Frontend → Vercel.** `public/` is plain static files:

```powershell
cd webapp
vercel    # or drag the public/ folder into the Vercel dashboard
```

Then point the deployed site at your backend (pick one):
1. Visit `https://your-app.vercel.app/?api=https://your-backend-url`
   (remembered in localStorage), or
2. Click your avatar (bottom-left) → **Set backend URL**, or
3. Hard-code it in `public/index.html` `<head>`:
   ```html
   <script>window.KARIMGPT_API = "https://your-backend-url";</script>
   ```

**Backend → somewhere that can run PyTorch.** Vercel can't (serverless size
limits), so the recommended free option is a **Hugging Face Space** (Docker
template): 16 GB RAM and always-on enough for all three models, free. Upload
the three weight files + tokenizers + `server.py`, set the `KARIMGPT_*_DIR`
env vars to where you put them, and you're done — the Space URL is your
backend (and it serves the UI too, so Vercel becomes optional). Render /
Railway / Fly.io also work, but their free tiers (~512 MB RAM) can only hold
the smaller models. For quick demos from your own PC: `ngrok http 8000`.

CORS is already open, so any frontend origin can call the backend.

---

## Notes
- Responses stream in like ChatGPT (pulsing cursor, markdown rendering, code
  blocks with copy buttons, regenerate, chat rename/search/history groups).
- Conversations live in your browser's `localStorage` — nothing is stored
  server-side. Each chat remembers which model it used.
- KarimGPT 1 is a pure text-continuation model (no chat training), so it
  *continues* whatever you type in Shakespearean style rather than answering.
- The models are small and trained from scratch, so output is often
  incoherent — that's expected. The point is the full ChatGPT experience
  running on *your own* models.
