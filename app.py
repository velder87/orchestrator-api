import os, json, requests, re
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -------- Config --------
AZ_SQL_SERVER   = os.getenv("AZ_SQL_SERVER")              # ex: myserver.database.windows.net
AZ_SQL_DB       = os.getenv("AZ_SQL_DB")
AZ_SQL_USER     = os.getenv("AZ_SQL_USER")                # ex: user@myserver
AZ_SQL_PASSWORD = os.getenv("AZ_SQL_PASSWORD")
TDS_VERSION     = os.getenv("TDS_VERSION", "7.3")         # pymssql supports up to 7.3 (Azure accepts it)

DBX_ENDPOINT_URL = os.getenv("DATABRICKS_ENDPOINT_URL", "").rstrip("/")  # .../serving-endpoints/<name>/invocations
DBX_TOKEN        = os.getenv("DATABRICKS_TOKEN", "")

# LLM — Azure OpenAI (recommandé) OU OpenAI classique (mets OPENAI_API_KEY)
AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY    = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")  # nom du déploiement (pas le modèle)
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY")  # si tu utilises OpenAI direct

CORS_ALLOW = os.getenv("CORS_ALLOW_ORIGINS", "*")

# -------- App --------
app = FastAPI(title="AI Agent Orchestrator", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ALLOW.split(",")] if CORS_ALLOW else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- DB engine (lazy) --------
_engine = None
def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    missing = [k for k,v in {
        "AZ_SQL_SERVER": AZ_SQL_SERVER, "AZ_SQL_DB": AZ_SQL_DB,
        "AZ_SQL_USER": AZ_SQL_USER, "AZ_SQL_PASSWORD": AZ_SQL_PASSWORD
    }.items() if not v]
    if missing:
        raise HTTPException(status_code=500, detail=f"DB not configured. Missing: {', '.join(missing)}")

    user_q = quote_plus(AZ_SQL_USER)
    pwd_q  = quote_plus(AZ_SQL_PASSWORD)
    # pymssql raises on unsupported TDS versions (7.4 is not accepted); cap to 7.3 which works with Azure SQL
    allowed_tds = {"7.0", "7.1", "7.2", "7.3"}
    tds = TDS_VERSION if TDS_VERSION in allowed_tds else "7.3"
    sql_url = f"mssql+pymssql://{user_q}:{pwd_q}@{AZ_SQL_SERVER}:1433/{AZ_SQL_DB}?tds_version={tds}&charset=utf8"

    _engine = create_engine(
        sql_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=int(os.getenv("SQL_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("SQL_MAX_OVERFLOW", "5")),
        connect_args={"timeout": 10}
    )
    return _engine

# -------- Tools --------
def guard_sql(q: str):
    """Reject non-read-only SQL statements."""

    # Reject multiple statements even if keywords are obfuscated with punctuation.
    # Example: "SELECT 1;DROP TABLE users" should be blocked.
    if ";" in q:
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are allowed.")

    banned_pattern = r"\b(drop|delete|update|insert|alter|create|truncate|merge)\b"
    if re.search(banned_pattern, q, flags=re.IGNORECASE):
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are allowed.")

def run_sql(query: str) -> Dict[str, Any]:
    guard_sql(query)
    try:
        eng = get_engine()
        with eng.begin() as conn:
            result = conn.execute(text(query))
            cols = list(result.keys())
            rows = result.fetchall()
        # pour le LLM: 2 formats
        as_list = [list(r) for r in rows]
        as_objs = [dict(zip(cols, list(r))) for r in rows]
        return {"columns": cols, "rows": as_list, "objects": as_objs}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"SQL error: {str(e)}") from e

def call_dbx_forecast(horizon: int = 6, product_id: Optional[int] = None) -> Dict[str, Any]:
    if not DBX_ENDPOINT_URL or not DBX_TOKEN:
        raise HTTPException(status_code=500, detail="Databricks endpoint not configured.")
    # Ton wrapper Prophet lit 'horizon' dans dataframe_records
    payload = {"dataframe_records": [{"horizon": int(horizon)}]}
    # Note: si tu ajoutes product_id à ton wrapper plus tard, insère-le ici.
    headers = {"Authorization": f"Bearer {DBX_TOKEN}", "Content-Type": "application/json"}
    r = requests.post(DBX_ENDPOINT_URL, headers=headers, json=payload, timeout=60)
    if r.status_code >= 300:
        raise HTTPException(status_code=500, detail=f"Databricks error: {r.text}")
    return r.json()  # souvent {"predictions":[...]} avec ds/yhat

# -------- LLM clients --------
def llm_chat(messages, tools=None) -> Dict[str, Any]:
    """
    Appelle Azure OpenAI (si configuré) sinon OpenAI. Retourne la réponse JSON (SDK HTTP minimal).
    """
    headers = {"Content-Type": "application/json"}
    body = {
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    # Azure OpenAI (Chat Completions)
    if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
        headers["api-key"] = AZURE_OPENAI_API_KEY
        url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=2024-08-01-preview"
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code >= 300:
            raise HTTPException(status_code=500, detail=f"Azure OpenAI error: {resp.text}")
        return resp.json()

    # OpenAI (fallback)
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        body["model"] = "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        if resp.status_code >= 300:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {resp.text}")
        return resp.json()

    raise HTTPException(status_code=500, detail="No LLM credentials configured.")

# -------- Tool schema (function calling) --------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Exécute une requête SELECT read-only sur Azure SQL et renvoie un tableau d'objets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requête SELECT (read-only)."},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_dbx_forecast",
            "description": "Appelle le modèle Databricks Serving pour générer des prévisions. Le wrapper attend un 'horizon'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "integer", "description": "Nombre de semaines à prévoir", "minimum": 1, "default": 6},
                    "product_id": {"type": "integer", "description": "Identifiant produit (optionnel)"}
                },
                "required": ["horizon"]
            }
        }
    }
]

SYSTEM_PROMPT = """Tu es un AI Agent orchestrateur.
- Comprends la question métier de l'utilisateur.
- Choisis le(s) outil(s) à appeler:
  * run_sql(query) pour toute agrégation, lookup, top-N sur les ventes (SELECT only).
  * call_dbx_forecast(horizon, product_id?) pour prédire les ventes futures.
- Après les outils, rédige une réponse PROPRE pour un décideur:
  * titre court (H2), 3 à 6 puces claires
  * si résultats tabulaires: mets un tableau en Markdown
  * si prévisions: résume les 2-3 points clés (tendance, intervalle, produits concernés)
  * conclue par une recommandation actionnable (1 phrase)
- Ne montre jamais de secrets. Si une action est impossible, explique précisément quoi faire (config manquante, droits, etc.)."""

# -------- Schemas --------
class ChatIn(BaseModel):
    message: str

class HealthOut(BaseModel):
    status: str
    db_configured: bool
    dbx_configured: bool
    llm: str

# -------- Endpoints --------
@app.get("/health", response_model=HealthOut)
def health():
    db_ok  = all([AZ_SQL_SERVER, AZ_SQL_DB, AZ_SQL_USER, AZ_SQL_PASSWORD])
    dbx_ok = bool(DBX_ENDPOINT_URL and DBX_TOKEN)
    llm = "azure" if (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY) else ("openai" if OPENAI_API_KEY else "none")
    return HealthOut(status="ok", db_configured=db_ok, dbx_configured=dbx_ok, llm=llm)

@app.post("/agent")
def agent_chat(req: ChatIn):
    user_msg = req.message.strip()

    # Raccourcis manuels
    if user_msg.lower().startswith("sql:"):
        data = run_sql(user_msg[4:].strip())
        # formatage simple
        return {
            "answer_markdown": "## Résultat SQL\n\n" + pd.DataFrame(data["objects"]).to_markdown(index=False),
            "tool": "run_sql",
            "data": data
        }

    # Boucle outils LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg}
    ]

    # 1er appel: voir si le LLM veut appeler un outil
    first = llm_chat(messages, tools=TOOLS)
    choice = first["choices"][0]["message"]

    # S'il n'y a pas d'appel d'outil, renvoyer la réponse directe
    tool_calls = choice.get("tool_calls", [])
    if not tool_calls:
        return {"answer_markdown": choice.get("content", "(vide)"), "data": None}

    # Exécuter les outils demandés (un ou plusieurs), collecter les résultats
    tool_results_msgs = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        args = json.loads(tc["function"].get("arguments", "{}"))

        if name == "run_sql":
            res = run_sql(args["query"])
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps(res, ensure_ascii=False)
            })
        elif name == "call_dbx_forecast":
            res = call_dbx_forecast(horizon=int(args.get("horizon", 6)), product_id=args.get("product_id"))
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps(res, ensure_ascii=False)
            })
        else:
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps({"error":"unknown tool"})
            })

    # 2e appel: donner les résultats au LLM pour mise en forme finale
    messages.append(choice)             # assistant avec tool_calls
    messages.extend(tool_results_msgs)  # résultats outillés
    second = llm_chat(messages, tools=None)
    final_answer = second["choices"][0]["message"]["content"]

    # Renvoyer aussi les bruts (pour debug UI)
    raw_payloads = [json.loads(m["content"]) for m in tool_results_msgs]
    return {
        "answer_markdown": final_answer,
        "tools_used": [tc["function"]["name"] for tc in tool_calls],
        "payloads": raw_payloads
    }
