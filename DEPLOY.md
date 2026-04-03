# Deploying to Hugging Face Spaces — Step-by-Step

This guide walks you through deploying the SQL Repair environment
to Hugging Face Spaces so it satisfies the hackathon's automated
validation (HF Space deploys, /health returns 200, /reset responds).

---

## Prerequisites

- A Hugging Face account (free): https://huggingface.co/join
- Git installed locally
- Your project folder: sql-repair-env/

---

## Step 1 — Install the HF CLI

```bash
pip install huggingface_hub
huggingface-cli login
#paste HF token when prompted
#token at: https://huggingface.co/settings/tokens
```

---

## Step 2 — Create a new Space

Go to: https://huggingface.co/new-space

Fill in:
  - Owner: your username
  - Space name: sql-repair-env
  - License: MIT
  - SDK: **Docker**   ← important, select Docker not Gradio/Streamlit
  - Hardware: CPU basic (free)  ← sufficient for this env

Click "Create Space".

You'll land on a page showing an empty Space.

---

## Step 3 — Add the openenv tag

In your Space settings (Settings tab → Tags):
Add the tag:  openenv

This is required for the hackathon's automated discovery.

---

## Step 4 — Clone the Space repo locally

```bash
#replace YOUR_USERNAME with your HF username
git clone https://huggingface.co/spaces/iambhavyaa/sql-repair-env
cd sql-repair-env
```

---

## Step 5 — Copy your project files into the cloned repo

```bash
# From the parent directory of both folders:
cp -r ../sql-repair-env/* .

# The folder looks like:
# .
# ├── app/
# │   ├── __init__.py
# │   ├── main.py
# │   ├── env.py
# │   ├── tasks.py
# │   ├── models.py
# │   ├── rewards.py
# │   └── graders.py
# ├── baseline.py
# ├── train_grpo.py
# ├── tests.py
# ├── openenv.yaml
# ├── Dockerfile
# ├── inference.py
# ├── requirements.txt
# └── README.md
```

---

## Step 6 — Add your OPENAI_API_KEY as a Space secret

In your Space on HF (Settings tab → Repository secrets):

  Name:  OPENAI_API_KEY
  Value: sk-...your key...

Click "Add new secret".

This lets /baseline run the real agent. Without it, /baseline runs
in demo mode (still returns 200, but uses pre-written queries).

---

## Step 7 — Push to HF

```bash
git add .
git commit -m "Initial deployment: SQL Repair OpenEnv"
git push
```

HF Spaces will automatically:
  1. Detect the Dockerfile
  2. Run: docker build
  3. Run: docker run -p 7860:7860
  4. Expose your Space at: https://iambhavyaa-sql-repair-env.hf.space

The build takes ~2-3 minutes. Watch the build logs in the "Logs" tab.

---

## Step 8 — Verify deployment

Once the Space shows "Running" (green dot):

```bash
BASE=https://iambhavyaa-sql-repair-env.hf.space

# 1. Health check
curl $BASE/health
# Expected: {"status":"ok","environment":"sql-repair-env"}

# 2. List tasks
curl $BASE/tasks
# Expected: JSON with 3 tasks

# 3. Reset + step (full round trip)
curl -X POST $BASE/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy"}'

curl -X POST $BASE/step \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT first_name, last_name, salary FROM employees WHERE department = '\''Engineering'\'' ORDER BY salary DESC;"}'

# 4. Baseline scores
curl $BASE/baseline
# Expected: average_score: 1.0 in demo mode, or real scores with API key
```

All four must succeed for the hackathon pre-submission checklist to pass.

---

## Step 9 — Submit

In the hackathon submission form, enter:
  - HF Space URL: https://huggingface.co/spaces/iambhavyaa/sql-repair-env
  - GitHub repo URL: (your repo if you have one)

---

## Troubleshooting

**Build fails with "port already in use"**
→ Make sure CMD in Dockerfile uses port 7860 (it does).

**Space shows "Error" after build**
→ Click "Logs" → "Container logs" and look for the Python traceback.
→ Most common: a missing import. Check requirements.txt has all deps.

**OPENAI_API_KEY not found**
→ /baseline will run in demo mode — that's fine for the submission check.
→ For real baseline scores: add the secret in Space Settings.

**Space keeps restarting**
→ The healthcheck is pinging /health every 30s. If your app crashes
   on startup, check logs. Most common cause: syntax error in Python files.
   Run `python -m py_compile app/main.py` locally to catch this early.

---

## After deployment — run the full test suite locally

```bash
cd sql-repair-env
pip install -r requirements.txt
pytest tests.py -v

# Or without pytest:
python tests.py
```

All 19 tests should pass before submission.
