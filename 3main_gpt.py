# ============================================
# FAST GRADIO APP USING PRECOMPUTED DATA
# ============================================
import os
import re
import json
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import pandas as pd
import gensim.downloader as gensim_api
from openai import OpenAI
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()

DATA_DIR = Path(os.getenv("KEYWORD_DATA_DIR", Path.cwd()))
TOP_PER_FILE = 1000
TOP_SIMILARITY_LIMIT = 1000
TOP_RANK_LIMIT = 100
TOP_FINAL_LIMIT = 20

WORD2VEC_MODEL_NAME = os.getenv("WORD2VEC_MODEL", "glove-wiki-gigaword-100")

PREP_PARQUET = DATA_DIR / "preprocessed_keywords.parquet"
PREP_EMB_NPY = DATA_DIR / "keyword_embeddings.npy"
PREP_NORMS_NPY = DATA_DIR / "keyword_norms.npy"
WORD_FREQ_PARQUET = DATA_DIR / "word_freq.parquet"
LEGACY_WORD_FREQ_CSV = DATA_DIR / "word_freq.csv"

CACHE = {
    "df": None,
    "emb_matrix": None,
    "emb_norms": None,
    "word_freq": None,
    "model": None,
    "initialized": False,
}

_gpt_client = None


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return re.findall(r"\b\w+\b", text.lower())


# ---------- LOADING PRECOMPUTED DATA ----------
def load_preprocessed():
    if CACHE["df"] is not None:
        return CACHE["df"], CACHE["emb_matrix"], CACHE["emb_norms"], CACHE["word_freq"]

    if not PREP_PARQUET.exists():
        raise RuntimeError(f"Preprocessed parquet not found: {PREP_PARQUET}")
    if not PREP_EMB_NPY.exists():
        raise RuntimeError(f"Embeddings npy not found: {PREP_EMB_NPY}")
    if not PREP_NORMS_NPY.exists():
        raise RuntimeError(f"Norms npy not found: {PREP_NORMS_NPY}")
    word_freq_path = None
    if WORD_FREQ_PARQUET.exists():
        word_freq_path = WORD_FREQ_PARQUET
    elif LEGACY_WORD_FREQ_CSV.exists():
        word_freq_path = LEGACY_WORD_FREQ_CSV
        print(
            f"Warning: {WORD_FREQ_PARQUET.name} not found, falling back to legacy CSV ({LEGACY_WORD_FREQ_CSV.name})."
        )
    if word_freq_path is None:
        raise RuntimeError(
            f"Word frequency file not found: expected {WORD_FREQ_PARQUET} (preferred) "
            f"or {LEGACY_WORD_FREQ_CSV}."
        )

    print("Loading preprocessed dataset & embeddings...")
    df = pd.read_parquet(PREP_PARQUET)
    emb_matrix = np.load(PREP_EMB_NPY)
    emb_norms = np.load(PREP_NORMS_NPY)
    if word_freq_path.suffix.lower() == ".parquet":
        word_freq_df = pd.read_parquet(word_freq_path)
    else:
        word_freq_df = pd.read_csv(word_freq_path)

    CACHE["df"] = df
    CACHE["emb_matrix"] = emb_matrix
    CACHE["emb_norms"] = emb_norms
    CACHE["word_freq"] = word_freq_df

    return df, emb_matrix, emb_norms, word_freq_df


def load_word2vec_model():
    if CACHE["model"] is not None:
        return CACHE["model"]
    print(f"Loading pretrained embeddings: {WORD2VEC_MODEL_NAME}")
    model = gensim_api.load(WORD2VEC_MODEL_NAME)
    CACHE["model"] = model
    return model


def initialize_once():
    """
    Preload datasets and embeddings before the UI starts so the first request
    doesn't block and so we can fail fast if files are missing.
    """
    if CACHE["initialized"]:
        return
    load_preprocessed()
    load_word2vec_model()
    CACHE["initialized"] = True
    print("Initialization complete. Ready to accept requests.")


def sentence_vector(tokens: List[str], model) -> np.ndarray | None:
    if not tokens:
        return None
    kv = getattr(model, "wv", model)
    vecs = [kv[w] for w in tokens if w in kv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def get_gpt_client():
    global _gpt_client
    if _gpt_client:
        return _gpt_client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found; skipping GPT relevance filter.")
        return None
    _gpt_client = OpenAI(api_key=api_key)
    return _gpt_client


def filter_with_gpt(user_keyword: str, ranked_df: pd.DataFrame) -> pd.DataFrame:
    client = get_gpt_client()
    if client is None or ranked_df.empty:
        return ranked_df

    candidates = [
        {"rank": int(row["MSC Rank"]), "keyword": str(row["searchTerm"])}
        for _, row in ranked_df.iterrows()
    ]
    prompt = (
        f"User keyword: {user_keyword}\n"
        "Ranked candidate keywords:\n"
        + "\n".join(f"{c['rank']}. {c['keyword']}" for c in candidates)
        + "\nReturn JSON with keep_ranks (list of ranking numbers to keep). "
          "Keep only items matching the same search intent. Do not invent ranks."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Filter out keywords that are off-topic; keep ranks that match the user intent. Return only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        message = response.choices[0].message.content
        keep_ranks = json.loads(message).get("keep_ranks", [])
        valid_ranks = {c["rank"] for c in candidates}
        keep_ranks = [
            int(rank)
            for rank in keep_ranks
            if isinstance(rank, (int, str)) and str(rank).isdigit() and int(rank) in valid_ranks
        ]
        if not keep_ranks:
            return ranked_df
        filtered = ranked_df[ranked_df["MSC Rank"].isin(keep_ranks)]
        return filtered.sort_values("MSC Rank")
    except Exception as exc:
        print(f"GPT relevance filter failed: {exc}")
        return ranked_df


# ---------- CORE METRICS ----------
def compute_similarity_views(keyword_vector: np.ndarray, df: pd.DataFrame,
                             emb_matrix: np.ndarray, emb_norms: np.ndarray) -> pd.DataFrame:
    # Vectorized cosine similarity: sim = (E·v) / (||E|| * ||v||)
    v = keyword_vector.astype(np.float32)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return pd.DataFrame()

    sims = emb_matrix.dot(v) / (emb_norms * v_norm)
    working = df.copy()
    working["searchTerm"] = working["searchTerm"].astype(str).str.strip()
    working["searchTerm_norm"] = working["searchTerm"].str.lower()
    working["similarity"] = sims

    subsets = []
    for file_name, group in working.groupby("file_name"):
        sorted_group = group.sort_values("similarity", ascending=False)
        subset = sorted_group.drop_duplicates(
            subset="searchTerm_norm", keep="first"
        ).head(TOP_PER_FILE)
        subset = subset.drop(columns=["searchTerm_norm"])
        subsets.append(subset)

    if not subsets:
        return pd.DataFrame()
    combined = pd.concat(subsets, ignore_index=True)
    combined = combined.sort_values("similarity", ascending=False).reset_index(drop=True)
    return combined


def compute_rank_table(similarity_df: pd.DataFrame) -> pd.DataFrame:
    if similarity_df.empty:
        return pd.DataFrame(
            columns=[
                "searchTerm",
                "repetition",
                "Normalized Repetition",
                "Normalized Rank Score",
                "MSC Score",
                "MSC Rank",
            ]
        )

    working = similarity_df.copy()
    working["searchTerm"] = working["searchTerm"].astype(str).str.strip()
    working["searchTerm_norm"] = working["searchTerm"].str.lower()
    working["inv_rank"] = 1 / working["searchFrequencyRank"]
    working = working.sort_values("similarity", ascending=False)
    grouped = (
        working.groupby("searchTerm_norm")
        .agg(
            searchTerm=("searchTerm", "first"),
            average_inv_rank=("inv_rank", "mean"),
            repetition=("searchTerm", "count"),
            best_similarity=("similarity", "max"),
            sample_file=("file_name", "first"),
        )
        .reset_index(drop=True)
    )

    max_inv_rank = grouped["average_inv_rank"].max()
    max_repetition = grouped["repetition"].max()

    grouped["Normalized Rank Score"] = (
        grouped["average_inv_rank"] / max_inv_rank * 100 if max_inv_rank else 0
    )
    grouped["Normalized Repetition"] = (
        grouped["repetition"] / max_repetition * 100 if max_repetition else 0
    )
    grouped["MSC Score"] = (
        0.7 * grouped["Normalized Repetition"] + 0.3 * grouped["Normalized Rank Score"]
    )
    grouped = grouped.sort_values("MSC Score", ascending=False).reset_index(drop=True)
    grouped["MSC Rank"] = range(1, len(grouped) + 1)
    return grouped


def build_word_counts(top20_df: pd.DataFrame, word_freq_df: pd.DataFrame) -> pd.DataFrame:
    if top20_df.empty:
        return pd.DataFrame(columns=["word", "count"])

    words = []
    for term in top20_df["searchTerm"]:
        words.extend(tokenize(term))

    unique_words = sorted(set(words))
    wf_map = dict(zip(word_freq_df["word"], word_freq_df["count"]))
    rows = [{"word": w, "count": wf_map.get(w, 0)} for w in unique_words]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# ---------- MAIN PIPELINE ----------
def process_keyword(keyword: str):
    keyword = (keyword or "").strip()
    if not keyword:
        msg = pd.DataFrame([{"message": "Please enter a keyword"}])
        return msg, msg, msg, pd.DataFrame()

    df, emb_matrix, emb_norms, word_freq_df = load_preprocessed()
    model = load_word2vec_model()

    keyword_tokens = tokenize(keyword)
    keyword_vec = sentence_vector(keyword_tokens, model)
    if keyword_vec is None:
        msg = pd.DataFrame([{"message": "Unable to build vector for the provided keyword."}])
        return msg, msg, msg, pd.DataFrame()

    similarity_df = compute_similarity_views(keyword_vec, df, emb_matrix, emb_norms)
    if similarity_df.empty:
        msg = pd.DataFrame([{"message": "No matching data found in Excel files."}])
        return msg, msg, msg, pd.DataFrame()

    similarity_view = similarity_df.head(TOP_SIMILARITY_LIMIT)[
        ["searchTerm", "similarity", "searchFrequencyRank", "file_name", "reportDate"]
    ]

    rank_table = compute_rank_table(similarity_df)
    top_ranked = rank_table.head(TOP_RANK_LIMIT)
    ranked_display = top_ranked[
        [
            "MSC Rank",
            "searchTerm",
            "MSC Score",
            "Normalized Rank Score",
            "Normalized Repetition",
            "repetition",
        ]
    ]

    filtered = filter_with_gpt(keyword, top_ranked)
    filtered = filtered.sort_values("MSC Score", ascending=False).reset_index(drop=True)
    filtered["Final Rank"] = range(1, len(filtered) + 1)
    final_top20 = filtered.head(TOP_FINAL_LIMIT)[
        [
            "Final Rank",
            "searchTerm",
            "MSC Score",
            "Normalized Rank Score",
            "Normalized Repetition",
            "repetition",
        ]
    ]

    word_counts = build_word_counts(final_top20, word_freq_df)

    return similarity_view, ranked_display, final_top20, word_counts


# ---------- GRADIO UI ----------
with gr.Blocks() as app:
    gr.Markdown("## Multi-day Keyword Ranking (FAST Version: Precomputed Embeddings)")
    gr.Markdown(
        "Enter a keyword to compare against all preprocessed Excel files. "
        "The heavy steps (tokenization, embeddings, merging) were done offline, "
        "so this runs much faster."
    )

    keyword_input = gr.Textbox(label="Enter Keyword")

    top_similarity_df = gr.DataFrame(
        label="Top results per file (first 1000 rows)",
        interactive=False,
    )
    top_ranked_df = gr.DataFrame(
        label="Top 100 after MSC ranking",
        interactive=False,
    )
    final_top20_df = gr.DataFrame(
        label="Final Top 20 (post-GPT)",
        interactive=False,
    )
    word_count_df = gr.DataFrame(
        label="Word repetition counts for Top 20",
        interactive=False,
    )

    run_btn = gr.Button("Run Analysis")
    run_btn.click(
        fn=process_keyword,
        inputs=keyword_input,
        outputs=[top_similarity_df, top_ranked_df, final_top20_df, word_count_df],
    )

if __name__ == "__main__":
    try:
        initialize_once()
        app.launch()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down the server gracefully.")
