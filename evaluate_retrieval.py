# evaluate_retrieval_fast.py

"""
Fast retrieval evaluation:

- Uses TF-IDF to retrieve top-K candidates (K=100 by default)

- Re-ranks those with the trained classifier (model_lr.joblib or model_rf.joblib)

- Computes Top-1, Top-3, Top-5 and MRR

Requirements: files produced by prepare_pairs.py and train_model.py:

- lost_found_dataset_realistic_metadata_30k.csv

- lost_found_exact_match_pairs.csv

- tfidf.joblib

- precomputed_matrices.npz

- scaler.joblib

- model_lr.joblib  (or model_rf.joblib)

"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from tqdm import tqdm
import time

# CONFIG
DATA_CSV = "lost_found_dataset_realistic_metadata_30k.csv"
EVAL_CSV = "lost_found_exact_match_pairs.csv"
TFIDF_PKL = "tfidf.joblib"
MATRIX_NPZ = "precomputed_matrices.npz"
SCALER_PKL = "scaler.joblib"
MODEL_PKL = "model_lr.joblib"   # switch to model_rf.joblib if you prefer RF
TOPK = 100   # top-k candidates from TF-IDF to re-rank

def jaccard_tokens(a, b):
    sa = set(str(a).lower().split())
    sb = set(str(b).lower().split())
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb)) / len(sa | sb)

def time_diff_hours(t1, t2):
    try:
        if pd.isna(t1) or pd.isna(t2):
            return 99999.0
        t1 = pd.to_datetime(t1, errors='coerce')
        t2 = pd.to_datetime(t2, errors='coerce')
        if pd.isna(t1) or pd.isna(t2):
            return 99999.0
        return abs((t1 - t2).total_seconds()) / 3600.0
    except Exception:
        return 99999.0

def build_features_for_candidates(df, tfidf_matrix, lost_idx, candidate_indices, cosines_subset=None):
    """
    Build feature DataFrame for the pairs (lost_idx, each candidate_index)
    cosines_subset: optional precomputed cosine values aligned with candidate_indices
    """
    lost_row = df.loc[lost_idx]
    # compute cosine if not provided
    if cosines_subset is None:
        cosines_subset = cosine_similarity(tfidf_matrix[lost_idx], tfidf_matrix[candidate_indices]).flatten()

    feats = []
    for i, fi in enumerate(candidate_indices):
        found = df.loc[fi]
        feat = {
            'cosine_text': float(cosines_subset[i]),
            'jaccard_desc': jaccard_tokens(lost_row['description'], found['description']),
            'color_match': 1 if str(lost_row.get('color','')).strip().lower()==str(found.get('color','')).strip().lower() else 0,
            'brand_match': 1 if str(lost_row.get('brand','')).strip().lower()==str(found.get('brand','')).strip().lower() else 0,
            'location_match': 1 if str(lost_row.get('location','')).strip().lower()==str(found.get('location','')).strip().lower() else 0,
            'user_match': 1 if str(lost_row.get('user','')).strip().lower()==str(found.get('user','')).strip().lower() else 0,
            'time_diff_hours': time_diff_hours(lost_row.get('timestamp', None), found.get('timestamp', None)),
            'len_diff': abs(len(str(lost_row['description'])) - len(str(found['description']))),
            'name_match': 1 if str(lost_row['item_name']).strip().lower()==str(found['item_name']).strip().lower() else 0
        }
        feats.append(feat)
    feats_df = pd.DataFrame(feats)
    return feats_df

def main():
    print("Loading data and artifacts...")
    df = pd.read_csv(DATA_CSV).reset_index(drop=True)
    eval_pairs = pd.read_csv(EVAL_CSV)
    tf = joblib.load(TFIDF_PKL)
    tfidf_matrix = sparse.load_npz(MATRIX_NPZ)
    scaler = joblib.load(SCALER_PKL)
    model = joblib.load(MODEL_PKL)

    # ensure timestamp parsed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    else:
        df['timestamp'] = pd.NaT

    # prepare mapping from file item_id -> df index (robust)
    id_to_idx_map = {}
    if 'item_id' in df.columns:
        for idx, row in df.iterrows():
            id_to_idx_map[int(row['item_id'])] = idx

    # build list of found indices (df row indices)
    found_indices = df[df['lost_or_found']=='found'].index.tolist()

    # build true_map: lost_df_index -> true_found_df_index
    true_map = {}
    for _, r in eval_pairs.iterrows():
        lid = int(r['lost_id'])
        fid = int(r['found_id'])
        if lid in id_to_idx_map and fid in id_to_idx_map:
            lost_idx = id_to_idx_map[lid]
            found_idx = id_to_idx_map[fid]
            true_map[lost_idx] = found_idx

    print(f"Total eval pairs to test: {len(true_map)}")

    topk_counters = {1:0, 3:0, 5:0}
    mrr_sum = 0.0
    n = 0
    t0 = time.time()

    # iterate over each lost item in true_map
    for lost_idx, true_found_idx in tqdm(list(true_map.items()), desc="Evaluating (fast)"):
        n += 1

        # get TF-IDF cosine to ALL found items (fast because sparse matrix dot)
        # compute as vector dot then normalize via sklearn's cosine_similarity for a single pair:
        # here using cosine_similarity for simplicity and reliability
        cos_all = cosine_similarity(tfidf_matrix[lost_idx], tfidf_matrix[found_indices]).flatten()

        # pick top-K candidate indices (indexes into found_indices)
        if len(cos_all) <= TOPK:
            top_pos = np.argsort(-cos_all)
        else:
            # argpartition for faster top-k
            top_pos = np.argpartition(-cos_all, TOPK-1)[:TOPK]
            # sort those top_pos
            top_pos = top_pos[np.argsort(-cos_all[top_pos])]

        candidate_found_indices = [found_indices[i] for i in top_pos]
        cos_subset = cos_all[top_pos]

        # build features for these candidates
        feats_df = build_features_for_candidates(df, tfidf_matrix, lost_idx, candidate_found_indices, cosines_subset=cos_subset)

        X = feats_df[[
            'cosine_text','jaccard_desc','color_match','brand_match',
            'location_match','user_match','time_diff_hours','len_diff','name_match'
        ]].fillna(0.0).astype(float)

        # scale & score
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[:,1]

        # rank candidates by prob desc
        ranked_idx_local = np.argsort(-probs)
        ranked_found_indices = [candidate_found_indices[i] for i in ranked_idx_local]

        # Top-k counters
        for k in topk_counters.keys():
            if true_found_idx in ranked_found_indices[:k]:
                topk_counters[k] += 1

        # MRR
        rank_pos = None
        for pos, fidx in enumerate(ranked_found_indices, start=1):
            if fidx == true_found_idx:
                rank_pos = pos
                break
        if rank_pos is not None:
            mrr_sum += 1.0 / rank_pos

        # periodic progress logging (every 500)
        if n % 500 == 0:
            elapsed = time.time() - t0
            rate = n / elapsed
            rem = (len(true_map) - n) / rate if rate > 0 else float('inf')
            print(f"[{n}/{len(true_map)}] elapsed={elapsed:.1f}s rate={rate:.2f}it/s est_rem={rem/60:.1f}min")

    # final metrics
    topk_scores = {k: topk_counters[k]/n for k in topk_counters}
    mrr = mrr_sum / n

    print(f"\nEvaluated on {n} pairs")
    for k,v in topk_scores.items():
        print(f"Top-{k} accuracy: {v:.4f}")
    print(f"MRR: {mrr:.4f}")

if __name__ == "__main__":
    main()
