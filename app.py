"""
app.py
------
Chat front-end. Multi-turn memory, a transparent view of every tool the agent
called, and an automated verification verdict on each answer.

Run locally:  streamlit run app.py
"""

import os
import streamlit as st

import config
import agent
import retrieval

st.set_page_config(page_title="Quant Research Copilot", page_icon="📈", layout="centered")

API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if API_KEY:
    os.environ["GROQ_API_KEY"] = API_KEY


@st.cache_resource(show_spinner="Building the vector knowledge base…")
def boot(key: str):
    """Create the client and warm up the vector DB once per session."""
    os.environ["GROQ_API_KEY"] = key
    client = agent.make_client(key)
    retrieval.prewarm()
    return client


st.title("📈 Quant Research Copilot")
st.caption("An AI agent that reasons over real cointegration tests, out-of-sample "
           "backtests, and a vector knowledge base — and verifies its own numbers.")

with st.sidebar:
    st.header("Architecture")
    st.markdown(
        "1. **Agent** (Groq Llama 3, tool-calling + memory)\n"
        "2. **Quant tools**: ADF + cointegration, OOS train/test backtest, "
        "pair screener, **portfolio** of pairs, **live long/short signal**\n"
        "3. **Vector DB** (Chroma + Local CPU embeddings): chunked company + methodology corpus, "
        "hybrid + metadata-filtered retrieval\n"
        "4. **Verifier** (LLM-as-judge, structured JSON output)\n"
    )
    st.divider()
    show_steps = st.toggle("Show agent steps", value=True)
    run_verify = st.toggle("Run answer verification", value=True)
    if st.button("Clear conversation"):
        st.session_state.pop("messages", None)
        st.session_state.pop("chat", None)
        st.rerun()
    st.divider()
    st.markdown("**Try:**\n"
                "- *Screen KO, PEP, XOM, CVX, JPM, BAC for pairs and explain the top one.*\n"
                "- *Backtest Visa vs Mastercard out-of-sample and tell me if it overfits.*\n"
                "- *Right now, which should I long/short — KO or PEP — and why?*\n"
                "- *Build a portfolio from KO,PEP,XOM,CVX,JPM,BAC,F,GM.*")
    if not API_KEY:
        st.error("No GROQ_API_KEY. Add it in Streamlit secrets or as an env var.")

if not API_KEY:
    st.stop()

client = boot(API_KEY)
if "chat" not in st.session_state:
    st.session_state.chat = agent.new_chat(client)
    st.session_state.messages = []

# Replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask a pairs-trading question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning, fetching data, running stats…"):
            try:
                answer, steps = agent.ask(st.session_state.chat, prompt)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        if show_steps and steps:
            with st.expander(f"🧠 Agent called {len(steps)} tool(s)"):
                for i, (name, args, result) in enumerate(steps, 1):
                    st.markdown(f"**{i}. `{name}`**")
                    st.write("args:", args)
                    st.write("result:", result)

        st.markdown(answer)

        if run_verify and steps:
            with st.spinner("Verifying numbers…"):
                v = agent.verify(client, prompt, answer, steps)
            grounded = v.get("all_numbers_grounded")
            conf = v.get("confidence")
            icon = "✅" if grounded else ("⚠️" if grounded is False else "ℹ️")
            conf_txt = f" · confidence {conf:.0%}" if isinstance(conf, (int, float)) else ""
            st.caption(f"{icon} Verifier: {v.get('one_line_verdict','')}{conf_txt}")
            if v.get("issues"):
                st.caption("Issues: " + "; ".join(v["issues"]))

    st.session_state.messages.append({"role": "assistant", "content": answer})
