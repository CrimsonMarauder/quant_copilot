"""
agent.py
--------
The "brain" and the architecture's coordination layer. Two stages:

  1. REASON & ACT - a Groq chat session (Llama 3) is given five tools. It plans,
     calls tools, reads results, and writes a grounded answer. Chat sessions
     keep multi-turn MEMORY.

  2. VERIFY - a second Groq call acts as an automated judge. Using JSON mode
     it checks that every number in the answer actually came from a tool result
     and returns a confidence score.
"""

from __future__ import annotations
import time
import json
from groq import Groq, InternalServerError, RateLimitError, APIConnectionError

import config
import quant_tools as qt
import retrieval

# -----------------------------------------------------------------------------
# Tool Definitions (OpenAI/Groq JSON Schema format)
# -----------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_pair",
            "description": "Run statistical tests (cointegration, ADF) to decide if two stocks are tradeable as a pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string", "description": "First ticker, e.g. 'KO'"},
                    "ticker2": {"type": "string", "description": "Second ticker, e.g. 'PEP'"},
                    "lookback_days": {"type": "integer", "description": "Days of history to analyze"}
                },
                "required": ["ticker1", "ticker2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_pair",
            "description": "Out-of-sample backtest of a z-score pairs strategy on two stocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string"},
                    "ticker2": {"type": "string"},
                    "lookback_days": {"type": "integer"},
                    "entry_z": {"type": "number", "description": "Z-score to open a trade"},
                    "exit_z": {"type": "number", "description": "Z-score to close a trade"}
                },
                "required": ["ticker1", "ticker2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "screen_pairs",
            "description": "Rank every pair in a basket of stocks by cointegration strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "string", "description": "Comma-separated tickers, e.g. 'KO,PEP,XOM'"},
                    "lookback_days": {"type": "integer"}
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_portfolio",
            "description": "Build and backtest a market-neutral portfolio from the cointegrated pairs in a basket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "string", "description": "Comma-separated tickers"},
                    "lookback_days": {"type": "integer"}
                },
                "required": ["tickers"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "current_signal",
            "description": "Give a live long/short recommendation for a pair NOW: which stock is rich vs cheap and which leg to long/short, with z-score, Bollinger bands, spread RSI, percentile, reversion trend and cautions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string", "description": "First ticker, e.g. 'KO'"},
                    "ticker2": {"type": "string", "description": "Second ticker, e.g. 'PEP'"},
                    "lookback_days": {"type": "integer"},
                    "entry_z": {"type": "number"},
                    "exit_z": {"type": "number"}
                },
                "required": ["ticker1", "ticker2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": "Search the vector database for text documents about companies or statistical methodology.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "scope": {"type": "string", "enum": ["all", "company", "knowledge"], "description": "Filter by type"}
                },
                "required": ["query"]
            }
        }
    }
]

# Map names to actual Python functions
_TOOL_FUNCTIONS = {
    "analyze_pair": qt.analyze_pair,
    "backtest_pair": qt.backtest_pair,
    "screen_pairs": qt.screen_pairs,
    "backtest_portfolio": qt.backtest_portfolio,
    "current_signal": qt.current_signal,
    "retrieve_context": retrieval.retrieve_context,
}

SYSTEM_PROMPT = """You are Quant Research Copilot. Your user is a smart investor who is NOT a statistician.
Your job is to give a clear, useful verdict and an actual recommendation — not a statistics lecture.

ALWAYS ANSWER IN THIS ORDER:
1. VERDICT (one line): Is this a good pair to trade right now? e.g.
   "Yes - tradeable, and there's a trade on right now." / "Tradeable, but no trade today." /
   "Not a good pair this period."
2. THE MOVE (one line): exactly what to do in plain words —
   "Buy X and short Y" or "No trade right now - wait." Name the stocks.
3. WHY (2-4 short sentences): translate the numbers into meaning, don't recite them.
4. RISKS (1-2 lines): what could go wrong / what would change the call.

PLAIN-LANGUAGE RULE — the first time you use a term, explain it in <=12 words in parentheses:
- cointegrated = "their prices move together, so the gap between them tends to snap back"
- spread = "the price gap between the two stocks"
- z-score = "how unusually wide that gap is right now"
- half-life = "roughly how many days the gap takes to close halfway"
- hedge ratio = "how many shares of one to trade per share of the other to stay balanced"
- mean-reversion = "the tendency to drift back to a normal level"
Never leave a piece of jargon unexplained. Prefer everyday words over finance words.

HARD RULES:
- Never state a number you did not get from a tool. Lead with the recommendation; use a few
  key numbers as support, translated into plain meaning (e.g. "the gap is wider than 95% of the
  past year, so it's stretched and likely to narrow").
- Use the tool's own plain_summary / plain_recommendation fields as your starting point.
- Be honest: if a pair is "weak" or "none", say it's not worth trading and suggest trying a
  longer history (e.g. "ask me to use 3 years") for two genuinely similar companies.
- Keep it short and skimmable. No tables of raw statistics. No restating every metric.

WORKFLOW:
- Always call analyze_pair first to get the verdict (it tests both directions and grades the pair).
- If tradeable, call current_signal for the live long/short call.
- Use backtest_pair to sanity-check, backtest_portfolio for a basket, screen_pairs to find candidates.
- Call retrieve_context (scope="company") to explain in plain words WHY the two move together.

End every answer with exactly: "Educational use only - not investment advice."
"""


def make_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def new_chat(client: Groq) -> list[dict]:
    """Start a chat session. Returns a list of messages."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except (InternalServerError, RateLimitError, APIConnectionError) as e:
            time.sleep(2 ** i + 1)
            if i == tries - 1:
                raise e
    raise RuntimeError("API unavailable after retries.")


def ask(messages: list[dict], question: str):
    """Send a question to Groq; handles the tool-calling loop. Returns (answer_text, steps)."""
    import os
    client = make_client(os.environ.get("GROQ_API_KEY", ""))
    
    # We mutate a copy of the conversation so we don't pollute the UI's history 
    # until the final answer is ready.
    temp_messages = list(messages)
    temp_messages.append({"role": "user", "content": question})
    
    steps = []
    
    # Tool calling loop (max 12 iterations)
    for _ in range(12):
        resp = _retry(lambda: client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=temp_messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        ))
        
        msg = resp.choices[0].message
        temp_messages.append(msg.model_dump(exclude_unset=True))
        
        # If the model didn't call any tools, we have our final answer!
        if not msg.tool_calls:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, steps
            
        # Otherwise, execute the tools
        for tc in msg.tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
                
            fn = _TOOL_FUNCTIONS.get(name)
            if fn:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {name}"}
                
            steps.append((name, args, result))
            
            # Send result back to Groq
            temp_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result)
            })
            
    return "Error: Agent exceeded maximum tool calls.", steps


def verify(client: Groq, question: str, answer: str, steps) -> dict:
    """LLM-as-judge: check the answer's numbers against tool outputs using Groq JSON mode."""
    tool_dump = "\n".join(f"{name}({args}) -> {result}" for name, args, result in steps) or "(no tools called)"
    prompt = (
        "You are auditing an AI answer for a pairs-trading question.\n\n"
        f"QUESTION:\n{question}\n\nTOOL OUTPUTS (the only valid source of numbers):\n{tool_dump}\n\n"
        f"ANSWER TO AUDIT:\n{answer}\n\n"
        "Check whether every numeric claim in the answer appears in the tool outputs. "
        "Flag any number that does not. Give a confidence in [0,1].\n\n"
        "You MUST return a raw JSON object with exactly these keys:\n"
        '{"all_numbers_grounded": boolean, "confidence": number, "issues": [list of strings], "one_line_verdict": string}'
    )
    
    try:
        resp = _retry(lambda: client.chat.completions.create(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        ))
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"all_numbers_grounded": None, "confidence": None,
                "issues": [f"verifier unavailable: {e}"], "one_line_verdict": "Verification skipped."}
