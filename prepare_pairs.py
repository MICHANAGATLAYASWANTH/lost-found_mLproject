#!/usr/bin/env python3
"""
prepare_pairs.py
Creates TF-IDF, precomputed TF-IDF matrix, and pairwise training data (pairs_train.pkl).

Usage:
    python prepare_pairs.py
    python prepare_pairs.py --data path/to/dataset.csv --pairs path/to/exact_pairs.csv --neg_per_pos 4
"""
import os
import random
from pathlib import Path
import argparse
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.sparse import save_npz
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------- Defaults (can be overridden by CLI) ----------
DATA_CSV = "lost_found_dataset_realistic_metadata_30k.csv"
EXACT_PAIRS_CSV = "lost_found_exact_match_pairs.csv"
TFIDF_JOBLIB = "tfidf.joblib"
PRECOMP_NPZ = "precomputed_matrices.npz"
PAIRS_PKL = "pairs_train.pkl"

# TF-IDF params
TFIDF_MAX_FEATURES = 30000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1

# Negative sampling: number of negatives per positive
NEG_PER_POS = 3

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------- Helpers ----------
def try_parse_time(x):
    if pd.isna(x):
        return np.nan
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
        "%Y/%m/%d %H:%M:%S",
    ]
    if isinstance(x, (int, float)) and not np.isnan(x):
        # maybe epoch seconds
        try:
            return datetime.fromtimestamp(int(x))
        except Exception:
            return np.nan
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    for f in fmts:
        try:
            return datetime.strptime(str(x), f)
        except Exception:
            continue
    try:
        return pd.to_datetime(x, errors="coerce").to_pydatetime()
    except Exception:
        return np.nan


def jaccard_tokens(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 0.0
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return len(inter) / len(union)


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)


# ---------- Main ----------
def main(args):
    print("Preparing pairs...")

    # --- 1) Load dataset ---
    if not Path(DATA_CSV).exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded dataset: {DATA_CSV} -> {len(df):,} rows")

    # ensure an index/id column exists
    if "id" in df.columns:
        id_col = "id"
    elif "item_id" in df.columns:
        id_col = "item_id"
    else:
        df = df.reset_index().rename(columns={"index": "id"})
        id_col = "id"
        print("No 'id' or 'item_id' column found. Created 'id' from dataframe index.")

    # normalize text columns presence
    desc_col = None
    if "description" in df.columns:
        desc_col = "description"
    elif "desc" in df.columns:
        desc_col = "desc"
    elif "item_description" in df.columns:
        desc_col = "item_description"
    else:
        raise ValueError("Description column not found (tried description/desc/item_description).")

    # optional metadata columns
    name_col = "item_name" if "item_name" in df.columns else ("name" if "name" in df.columns else None)
    color_col = "color" if "color" in df.columns else None
    brand_col = "brand" if "brand" in df.columns else None
    location_col = "location" if "location" in df.columns else None
    user_col = "user" if "user" in df.columns else None
    time_col = None
    for candidate in ["timestamp", "time", "created_at", "datetime"]:
        if candidate in df.columns:
            time_col = candidate
            break

    print("Detected columns:")
    print(f"  id: {id_col}")
    print(f"  description: {desc_col}")
    print(f"  name: {name_col}")
    print(f"  color: {color_col}")
    print(f"  brand: {brand_col}")
    print(f"  location: {location_col}")
    print(f"  user: {user_col}")
    print(f"  time: {time_col}")

    # --- 2) Fit or load TF-IDF ---
    if Path(TFIDF_JOBLIB).exists():
        print(f"Loading existing TF-IDF vectorizer from {TFIDF_JOBLIB}")
        tfidf = joblib.load(TFIDF_JOBLIB)
    else:
        print("Fitting new TF-IDF vectorizer...")
        corpus = df[desc_col].fillna("").astype(str).values
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF)
        tfidf.fit(corpus)
        joblib.dump(tfidf, TFIDF_JOBLIB)
        print(f"Saved TF-IDF to {TFIDF_JOBLIB}")

    # compute TF-IDF matrix for all items
    print("Transforming corpus into TF-IDF matrix...")
    tfidf_matrix = tfidf.transform(df[desc_col].fillna("").astype(str).values)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Saving TF-IDF matrix to {PRECOMP_NPZ} ...")
    save_npz(PRECOMP_NPZ, tfidf_matrix)
    print("Saved precomputed TF-IDF matrix.")

    # --- 3) Load exact pairs (positives) ---
    if not Path(EXACT_PAIRS_CSV).exists():
        raise FileNotFoundError(f"Exact pairs file not found: {EXACT_PAIRS_CSV}")
    pairs_df = pd.read_csv(EXACT_PAIRS_CSV)
    print(f"Loaded exact pairs: {EXACT_PAIRS_CSV} -> {len(pairs_df):,} rows")

    # detect columns in exact pairs file
    cols = pairs_df.columns.tolist()
    lost_col = None
    found_col = None
    for name in cols:
        if name.lower() in ("lost_id", "lostid", "lost"):
            lost_col = name
        if name.lower() in ("found_id", "foundid", "found"):
            found_col = name
    if lost_col is None or found_col is None:
        if len(cols) >= 2:
            lost_col, found_col = cols[0], cols[1]
            print(f"Using first two columns from {EXACT_PAIRS_CSV} as lost/found: {lost_col}, {found_col}")
        else:
            raise ValueError(f"Could not infer lost/found columns from {EXACT_PAIRS_CSV}: {cols}")

    print(f"Using columns in exact pairs: lost='{lost_col}', found='{found_col}'")

    # create mapping from id value -> integer row index in df (0..n-1)
    id_to_idx = {}
    for idx, val in enumerate(df[id_col].values):
        id_to_idx[val] = idx

    # try to handle mismatched types by adding stringified keys
    id_to_idx_str = {str(k): v for k, v in id_to_idx.items()}
    # merge: prefer original keys, but string keys available
    id_to_idx_full = dict(id_to_idx)
    id_to_idx_full.update(id_to_idx_str)

    # build list of positive pairs as (lost_idx, found_idx)
    pos_pairs = []
    missing_count = 0
    for _, r in pairs_df.iterrows():
        a = r[lost_col]
        b = r[found_col]
        ia = id_to_idx_full.get(a, id_to_idx_full.get(str(a), None))
        ib = id_to_idx_full.get(b, id_to_idx_full.get(str(b), None))
        if ia is None or ib is None:
            missing_count += 1
            continue
        pos_pairs.append((int(ia), int(ib)))
    if missing_count > 0:
        print(f"Warning: {missing_count} exact-pair rows referenced ids not found in dataset and were skipped.")
    print(f"Positive pairs mapped (kept): {len(pos_pairs):,}")

    # --- 4) Partition items into lost vs found if available ---
    lost_mask = None
    if "type" in df.columns:
        lost_mask = df["type"].astype(str).str.lower() == "lost"
        print("Using 'type' column to identify lost vs found items.")
    elif "is_lost" in df.columns:
        lost_mask = df["is_lost"].astype(bool)
        print("Using 'is_lost' column to identify lost vs found items.")
    else:
        lost_indices = set([a for a, _ in pos_pairs])
        found_indices = set([b for _, b in pos_pairs])
        if len(lost_indices) > 0:
            lost_mask = pd.Series(False, index=df.index)
            lost_mask.loc[list(lost_indices)] = True
            print("Inferred lost items from exact pairs left-column.")
        else:
            n = len(df)
            half = n // 2
            lost_mask = pd.Series([True if i < half else False for i in range(n)], index=df.index)
            print("No explicit lost/found info; defaulted to first half = lost, second half = found.")

    lost_indices_all = list(np.where(lost_mask.values)[0])
    found_indices_all = list(np.where(~lost_mask.values)[0])
    print(f"Total lost items detected: {len(lost_indices_all):,}")
    print(f"Total found items detected: {len(found_indices_all):,}")

    # --- 5) Negative sampling ---
    print(f"Sampling {NEG_PER_POS} negatives per positive (seed={RANDOM_SEED})...")
    neg_pairs = []
    found_set = set(found_indices_all)
    for (li, fi) in pos_pairs:
        choices = list(found_set - {fi})
        if not choices:
            continue
        n_to_sample = min(NEG_PER_POS, len(choices))
        sampled = random.sample(choices, n_to_sample)
        for s in sampled:
            neg_pairs.append((li, s))

    print(f"Negative pairs created: {len(neg_pairs):,}")

    # --- 6) Build DataFrame with features for each pair ---
    rows = []
    print("Computing features for pairs (this may take a bit)...")
    tfidf_mat = tfidf_matrix.tocsr()

    all_pair_items = [(*p, 1) for p in pos_pairs] + [(*p, 0) for p in neg_pairs]
    for i, j, label in tqdm(all_pair_items, desc="pairs", total=len(all_pair_items)):
        desc_i = safe_str(df.at[i, desc_col])
        desc_j = safe_str(df.at[j, desc_col])

        # cosine: use vector rows
        try:
            vec_i = tfidf_mat.getrow(i)
            vec_j = tfidf_mat.getrow(j)
            cos = 0.0
            if vec_i.nnz == 0 or vec_j.nnz == 0:
                cos = 0.0
            else:
                cos = float(cosine_similarity(vec_i, vec_j)[0, 0])
        except Exception:
            cos = 0.0

        jacc = jaccard_tokens(desc_i, desc_j)

        color_match = 0
        if color_col and not pd.isna(df.at[i, color_col]) and not pd.isna(df.at[j, color_col]):
            color_match = int(str(df.at[i, color_col]).strip().lower() == str(df.at[j, color_col]).strip().lower())

        brand_match = 0
        if brand_col and not pd.isna(df.at[i, brand_col]) and not pd.isna(df.at[j, brand_col]):
            brand_match = int(str(df.at[i, brand_col]).strip().lower() == str(df.at[j, brand_col]).strip().lower())

        location_match = 0
        if location_col and not pd.isna(df.at[i, location_col]) and not pd.isna(df.at[j, location_col]):
            location_match = int(str(df.at[i, location_col]).strip().lower() == str(df.at[j, location_col]).strip().lower())

        user_match = 0
        if user_col and not pd.isna(df.at[i, user_col]) and not pd.isna(df.at[j, user_col]):
            user_match = int(str(df.at[i, user_col]).strip().lower() == str(df.at[j, user_col]).strip().lower())

        name_match = 0
        if name_col and not pd.isna(df.at[i, name_col]) and not pd.isna(df.at[j, name_col]):
            name_match = int(str(df.at[i, name_col]).strip().lower() == str(df.at[j, name_col]).strip().lower())

        len_diff = abs(len(desc_i) - len(desc_j))

        # time difference in hours
        time_diff_hours = np.nan
        if time_col:
            t1 = try_parse_time(df.at[i, time_col])
            t2 = try_parse_time(df.at[j, time_col])
            if (t1 is not None) and (t2 is not None) and (not pd.isna(t1)) and (not pd.isna(t2)):
                try:
                    delta = abs((t1 - t2).total_seconds()) / 3600.0
                    time_diff_hours = float(delta)
                except Exception:
                    time_diff_hours = np.nan

        rows.append(
            {
                "lost_idx": int(i),
                "found_idx": int(j),
                "lost_id": df.at[i, id_col],
                "found_id": df.at[j, id_col],
                "cosine_text": cos,
                "jaccard_desc": jacc,
                "color_match": int(color_match),
                "brand_match": int(brand_match),
                "location_match": int(location_match),
                "user_match": int(user_match),
                "name_match": int(name_match),
                "len_diff": int(len_diff),
                "time_diff_hours": time_diff_hours if not pd.isna(time_diff_hours) else np.nan,
                "label": int(label),
            }
        )

    pairs_out = pd.DataFrame(rows)
    print(f"Built pairs dataframe: {pairs_out.shape[0]:,} rows x {pairs_out.shape[1]} cols")

    pairs_out = pairs_out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # save
    print(f"Saving pairs to {PAIRS_PKL} ...")
    pairs_out.to_pickle(PAIRS_PKL)
    print("Saved pairs pickle.")

    print("Done. Outputs:")
    print(f"  - {TFIDF_JOBLIB}")
    print(f"  - {PRECOMP_NPZ}")
    print(f"  - {PAIRS_PKL}")
    print("You can now run: python train_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_CSV, help="Dataset CSV path")
    parser.add_argument("--pairs", default=EXACT_PAIRS_CSV, help="Exact pairs CSV path")
    parser.add_argument("--neg_per_pos", type=int, default=NEG_PER_POS, help="Negatives per positive pair")
    args = parser.parse_args()

    # override globals if CLI args provided
    DATA_CSV = args.data
    EXACT_PAIRS_CSV = args.pairs
    NEG_PER_POS = args.neg_per_pos

    main(args)
