"""
config.py
---------
One place for all the knobs: which models to use, retrieval settings, and the
default trading-strategy parameters. Keeping these here (instead of scattered
through the code) is a small but real architecture improvement.
"""

# --- Models ---
CHAT_MODEL = "llama-3.3-70b-versatile"           # the reasoning / tool-calling agent (Groq)
JUDGE_MODEL = "llama-3.3-70b-versatile"          # the verification pass (Groq)
EMBED_MODEL = "all-MiniLM-L6-v2"         # turns text into vectors for the vector DB (Local CPU)

# --- Vector store ---
CHROMA_DIR = "chroma_db"                 # folder where the vector database persists
COLLECTION = "quant_knowledge"
TOP_K = 4                                # how many chunks to retrieve
HYBRID_ALPHA = 0.7                       # 1.0 = pure semantic, 0.0 = pure keyword

# --- Default strategy params (used by the quant tools) ---
LOOKBACK_DAYS = 365
TRAIN_FRAC = 0.7                         # fraction of history used to FIT, rest is out-of-sample
ROLL_WINDOW = 30                         # rolling window (days) for the z-score
ENTRY_Z = 2.0
EXIT_Z = 0.5
COINT_PVALUE = 0.10                      # tradeable cutoff (loosened from 0.05 to catch borderline twins)
COINT_STRONG = 0.05                      # below this = strong, high-confidence cointegration
COINT_WEAK = 0.20                        # between PVALUE and this = weak/loose relationship
