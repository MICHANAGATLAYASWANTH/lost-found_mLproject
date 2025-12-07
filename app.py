# app.py
"""
Campus Lost & Found — AutoMatch (Images)
Updated: removed ALL file upload (drag & drop) inputs; boosted confidence scoring.

Run with:
    pip install streamlit pandas numpy joblib scipy scikit-learn pillow requests
    # optional for CLIP-based image similarity (currently unused, but kept for future):
    pip install sentence-transformers
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import json, os, io, uuid, hashlib, re
from PIL import Image, ImageDraw, ImageFont
import requests

# Optional: sentence-transformers for CLIP embeddings (currently unused here, but kept for future)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------- CONFIG ----------------
DATA_CSV = "lost_found_dataset_realistic_metadata_30k_with_placeholders.csv"
CACHE_DIR = "image_cache"
PLACEHOLDER_DIR = "placeholders"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLACEHOLDER_DIR, exist_ok=True)

PL_SIZE = (420, 420)
PL_BG = (245, 247, 250)   # light placeholder background
PL_TEXT = (20, 20, 25)

IMAGE_EMB_NPZ = "image_embeddings_found.npz"  # kept for compatibility
FEEDBACK_JSON = "feedback.json"

# tuneables (presence + name boosts)
DEFAULT_TOPK_PREFILTER = 200
DEFAULT_ALPHA_TEXT = 0.6
PRESENCE_BOOST = 0.20     # boost when an image file exists (even without image similarity)
NAME_MATCH_BOOST = 0.12   # boost when item_name tokens match candidate item_name/category

# extra heuristic boosts
COLOR_BOOST = 0.10        # when color matches exactly
LOCATION_BOOST = 0.10     # when location hint matches location field
EXACT_NAME_BOOST = 0.25   # when item_name matches almost exactly

# ---------------- THEME (light modern) ----------------
st.set_page_config(page_title="Lost & Found — AutoMatch", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --muted: #6b7280;
      --accent: #0ea5a0;
      --accent-2: #0369a1;
    }
    body { background-color: var(--bg); color: #0f172a; }
    .app-title { font-family: "Inter", sans-serif; font-weight:700; font-size:28px; color:var(--accent-2); }
    .card { background: var(--card); padding:14px; border-radius:12px; box-shadow: 0 6px 18px rgba(16,24,40,0.06); }
    .muted { color: var(--muted); font-size:13px; }
    .small { font-size:13px; color:#374151; }
    .stButton>button { background-color: var(--accent-2); color: white; border-radius:8px; padding:8px 12px; }
    .stFileUploader>div { background: transparent; }
    .result-score { font-weight:700; color:var(--accent-2); font-size:20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ Helpers ------------------
def safe_filename(s: str) -> str:
    s = str(s)
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in s)[:200]

def placeholder_path_for_row(row: pd.Series) -> str:
    iid = row.get("item_id") or row.get("id") or row.get("uuid") or ""
    cat = row.get("category") or row.get("item_name") or "item"
    color = row.get("color") or ""
    base = f"{iid}_{cat}_{color}"
    return os.path.abspath(os.path.join(PLACEHOLDER_DIR, f"ph_{safe_filename(base)}.png"))

def generate_placeholder(row: pd.Series, force=False) -> str:
    out = placeholder_path_for_row(row)
    if os.path.exists(out) and not force:
        return out
    lines = []
    iid = row.get("item_id") or row.get("id") or row.get("uuid") or ""
    if iid:
        lines.append(f"ID: {iid}")
    cat = row.get("category") or row.get("item_name") or ""
    if cat:
        lines.append(" ".join(str(cat).split()[:3]))
    col = row.get("color") or ""
    if col:
        lines.append(f"Color: {col}")
    if not lines:
        lines = ["Lost item"]
    img = Image.new("RGB", PL_SIZE, PL_BG)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    text = "\n".join(lines)
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except Exception:
        w, h = draw.textsize(text, font=font)
    x = max(10, (PL_SIZE[0] - w) // 2)
    y = max(10, (PL_SIZE[1] - h) // 2)
    draw.multiline_text((x, y), text, fill=PL_TEXT, font=font, spacing=6)
    try:
        img.save(out, format="PNG")
    except Exception:
        out = os.path.abspath(os.path.join(PLACEHOLDER_DIR, f"ph_{uuid.uuid4().hex}.png"))
        img.save(out, format="PNG")
    return out

def cached_remote_image_path(url: str):
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
    p = os.path.join(CACHE_DIR, f"{key}{ext}")
    if os.path.exists(p):
        return os.path.abspath(p)
    try:
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        with open(p, "wb") as f:
            f.write(resp.content)
        return os.path.abspath(p)
    except Exception:
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass
        raise

def resolve_local_display_image(row: pd.Series):
    # 1) image_path field (absolute or relative)
    p = row.get("image_path") or row.get("photo_path")
    if isinstance(p, str) and p.strip():
        if os.path.isabs(p) and os.path.exists(p):
            return os.path.abspath(p)
        localp = os.path.abspath(p)
        if os.path.exists(localp):
            return localp
    # 2) image_url -> cached
    url = row.get("image_url")
    if isinstance(url, str) and url.strip():
        try:
            r = cached_remote_image_path(url)
            if os.path.exists(r):
                return r
        except Exception:
            pass
    # fallback placeholder
    return generate_placeholder(row)

# ---------------- load csv immediate (no caching) ----------------
def load_csv_now(path: str):
    if not os.path.exists(path):
        sample = []
        cats = ["Water Bottle", "Wallet", "Bag", "Phone", "Laptop", "Keys"]
        colors = ["black", "blue", "pink", "red", "white"]
        locs = ["Library", "Cafeteria", "Admin Block"]
        for i in range(1, 21):
            sample.append({
                "item_id": i,
                "item_name": cats[i % len(cats)],
                "category": cats[i % len(cats)],
                "description": f"{colors[i % len(colors)]} {cats[i % len(cats)]} found near {locs[i % len(locs)]}",
                "color": colors[i % len(colors)],
                "brand": "",
                "location": locs[i % len(locs)],
                "timestamp": datetime(2024, 1, 1).isoformat(),
                "image_path": "",
                "image_url": "",
                "lost_or_found": "found" if i % 2 == 0 else "lost",
                "user": f"User{i}"
            })
        df_sample = pd.DataFrame(sample)
        try:
            df_sample.to_csv(path, index=False)
        except:
            pass
        return df_sample
    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame()
    return df

# ---------------- optional artifacts (guarded) ----------------
def load_optional_artifacts():
    tf = tfidf_matrix = scaler = model = None
    image_embeddings = image_item_ids = image_paths = None
    embedder = None
    if os.path.exists("tfidf.joblib"):
        try:
            tf = joblib.load("tfidf.joblib")
        except:
            tf = None
    if os.path.exists("precomputed_matrices.npz"):
        try:
            tfidf_matrix = sparse.load_npz("precomputed_matrices.npz")
        except:
            tfidf_matrix = None
    if os.path.exists("scaler.joblib"):
        try:
            scaler = joblib.load("scaler.joblib")
        except:
            scaler = None
    if os.path.exists("model_lr.joblib"):
        try:
            model = joblib.load("model_lr.joblib")
        except:
            model = None
    if os.path.exists(IMAGE_EMB_NPZ):
        try:
            npz = np.load(IMAGE_EMB_NPZ, allow_pickle=True)
            image_embeddings = npz["embeddings"]
            image_item_ids = npz["item_ids"]
            image_paths = npz["image_paths"] if "image_paths" in npz else None
        except:
            image_embeddings = None
    if SentenceTransformer is not None:
        try:
            embedder = SentenceTransformer("clip-ViT-B-32")
        except:
            embedder = None
    return tf, tfidf_matrix, scaler, model, image_embeddings, image_item_ids, image_paths, embedder

# ---------------- init ----------------
if "df" not in st.session_state:
    st.session_state.df = load_csv_now(DATA_CSV)
df = st.session_state.df
tf, tfidf_matrix, scaler, model, image_embeddings, image_item_ids, image_paths, embedder = load_optional_artifacts()

# ensure columns
for col in ["item_name", "description", "color", "brand", "location"]:
    if col not in df.columns:
        df[col] = ""
if "text_blob" not in df.columns:
    df["text_blob"] = (
        df["item_name"].astype(str)
        + " "
        + df["description"].astype(str)
        + " "
        + df["color"].astype(str)
    ).str.strip()

# ---------------- UI ----------------
st.sidebar.title("Controls & Tuning")
st.sidebar.write("Rows:", df.shape[0])
st.sidebar.markdown("---")
topk = st.sidebar.slider("TF-IDF prefilter top-K", 50, 2000, DEFAULT_TOPK_PREFILTER, step=50)
alpha_text = st.sidebar.slider("Text weight (alpha)", 0.0, 1.0, DEFAULT_ALPHA_TEXT, step=0.05)
show_images_first = st.sidebar.checkbox("Prefer results that have images", value=True)
st.sidebar.markdown("---")
st.sidebar.write("Image embedder present:", "Yes" if embedder is not None else "No")
st.sidebar.write("Precomputed image embeddings present:", "Yes" if os.path.exists(IMAGE_EMB_NPZ) else "No")

# adjustable boosts
presence_boost_slider = st.sidebar.slider(
    "Presence boost (when image file exists)", 0.0, 0.7, PRESENCE_BOOST, step=0.01
)
name_match_boost_slider = st.sidebar.slider(
    "Name-match boost", 0.0, 0.4, NAME_MATCH_BOOST, step=0.01
)
color_boost_slider = st.sidebar.slider(
    "Color match boost", 0.0, 0.3, COLOR_BOOST, step=0.01
)
location_boost_slider = st.sidebar.slider(
    "Location match boost", 0.0, 0.3, LOCATION_BOOST, step=0.01
)
exact_name_boost_slider = st.sidebar.slider(
    "Exact name match boost", 0.0, 0.5, EXACT_NAME_BOOST, step=0.01
)

if st.sidebar.button("Clear image caches"):
    for d in [CACHE_DIR, PLACEHOLDER_DIR]:
        for fn in os.listdir(d):
            try:
                os.remove(os.path.join(d, fn))
            except:
                pass
    st.sidebar.success("Cleared caches")

st.markdown("<div class='app-title'>Campus Lost & Found — AutoMatch</div>", unsafe_allow_html=True)
tabs = st.tabs(["Search", "Diagnostics", "Manage"])

# ---------------- Search Tab ----------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Describe what you lost")
    col1, col2 = st.columns([2, 1])
    with col1:
        item_name = st.text_input("Item name (optional)")
        description = st.text_area("Description (include color/brand/marks)", height=140)
        color = st.text_input("Color (optional)")
        location_hint = st.text_input("Last seen location (optional)")
        search_btn = st.button("Find Matches")
    with col2:
        st.markdown("### Options")
        only_with_images = st.checkbox("Only show results with images", value=False)
        results_limit = st.slider("Max results to display", 1, 50, 12)
    st.markdown("</div>", unsafe_allow_html=True)

    if search_btn:
        if not description.strip() and not item_name.strip():
            st.error("Provide at least description or item name.")
        else:
            found_df = (
                df[df.get("lost_or_found", "found") == "found"].copy()
                if "lost_or_found" in df.columns
                else df.copy()
            )
            idxs = found_df.index.tolist()
            lost_blob = f"{item_name} {description} {color} {location_hint}".strip()

            # TF-IDF prefilter if available
            if tf is not None and tfidf_matrix is not None:
                try:
                    v = tf.transform([lost_blob])
                    cos_all = cosine_similarity(v, tfidf_matrix[idxs]).flatten()
                except Exception:
                    cos_all = np.zeros(len(idxs))
            else:
                tokens = set(lost_blob.lower().split())
                cos_all = np.zeros(len(idxs))
                for i, idxx in enumerate(idxs):
                    text = (
                        str(found_df.loc[idxx, "text_blob"])
                        if "text_blob" in found_df.columns
                        else ""
                    )
                    text_tokens = set(text.lower().split())
                    cos_all[i] = len(tokens & text_tokens) / max(
                        1, len(tokens | text_tokens)
                    )

            if len(cos_all) == 0:
                st.info("No found items available.")
            else:
                # pick top-K
                if len(cos_all) <= topk:
                    pos = np.argsort(-cos_all)
                else:
                    pos = np.argpartition(-cos_all, topk - 1)[:topk]
                    pos = pos[np.argsort(-cos_all[pos])]
                cand_indices = [idxs[i] for i in pos]
                cos_subset = cos_all[pos]
                cand_rows = df.loc[cand_indices].reset_index(drop=True)

                # no query image ⇒ no image similarity
                image_sims_for_candidates = None

                # scoring features
                feats = []
                desc_tokens = set(str(description).lower().split())
                for i, _ in enumerate(cand_indices):
                    row = cand_rows.iloc[i]
                    cand_desc_tokens = set(str(row.get("description", "")).lower().split())
                    jacc = len(desc_tokens & cand_desc_tokens) / max(
                        1, len(desc_tokens | cand_desc_tokens)
                    )
                    feat = {
                        "cos_text": float(cos_subset[i]) if i < len(cos_subset) else 0.0,
                        "jaccard": jacc,
                    }
                    feats.append(feat)
                feats_df = pd.DataFrame(feats).fillna(0.0)

                # combine TF-IDF and Jaccard for stronger text signal
                combined_text = 0.7 * feats_df["cos_text"].values + 0.3 * feats_df["jaccard"].values
                text_probs = combined_text

                # image contribution is zero (no upload), but we keep structure
                if image_sims_for_candidates is not None:
                    img_sim = np.clip(image_sims_for_candidates, -1.0, 1.0)
                    img_sim_norm = (img_sim + 1.0) / 2.0
                else:
                    img_sim_norm = np.zeros_like(text_probs)

                # presence boost when an actual image file exists
                presence_flags = np.array(
                    [os.path.exists(str(row.get("image_path") or "")) for _, row in cand_rows.iterrows()],
                    dtype=float,
                )
                presence_boost_arr = np.where(
                    presence_flags > 0,
                    float(presence_boost_slider),
                    0.0,
                )

                # name match boost: user item_name tokens appear in candidate name/category
                name_tokens = (
                    set(
                        [
                            t
                            for t in re.split(r"[\W_]+", item_name.lower())
                            if len(t) > 1
                        ]
                    )
                    if item_name.strip()
                    else set()
                )
                name_boost_arr = np.zeros_like(text_probs)
                exact_name_boost_arr = np.zeros_like(text_probs)
                color_boost_arr = np.zeros_like(text_probs)
                location_boost_arr = np.zeros_like(text_probs)

                item_name_clean = item_name.strip().lower()
                color_clean = color.strip().lower()
                loc_hint_clean = location_hint.strip().lower()

                for i in range(len(cand_rows)):
                    r = cand_rows.iloc[i]
                    cand_name = str(r.get("item_name", "")).lower()
                    cand_cat = str(r.get("category", "")).lower()
                    cand_color = str(r.get("color", "")).lower()
                    cand_loc = str(r.get("location", "")).lower()

                    target_text = f"{cand_name} {cand_cat}"

                    # token-based name match
                    if name_tokens and any(t in target_text for t in name_tokens):
                        name_boost_arr[i] = float(name_match_boost_slider)

                    # almost exact name match (ignoring case and some punctuation)
                    if item_name_clean:
                        if item_name_clean == cand_name or item_name_clean in cand_name or cand_name in item_name_clean:
                            exact_name_boost_arr[i] = float(exact_name_boost_slider)

                    # color match
                    if color_clean and color_clean == cand_color:
                        color_boost_arr[i] = float(color_boost_slider)

                    # location match (hint appears in location)
                    if loc_hint_clean and loc_hint_clean in cand_loc:
                        location_boost_arr[i] = float(location_boost_slider)

                # final score
                final_scores = (
                    (alpha_text * text_probs)
                    + ((1.0 - alpha_text) * img_sim_norm)
                    + presence_boost_arr
                    + name_boost_arr
                    + exact_name_boost_arr
                    + color_boost_arr
                    + location_boost_arr
                )
                # clamp to [0,1]
                final_scores = np.clip(final_scores, 0.0, 1.0)

                cand_rows = cand_rows.copy()
                cand_rows["cos_text"] = feats_df["cos_text"].values
                cand_rows["jaccard"] = feats_df["jaccard"].values
                cand_rows["text_prob"] = text_probs
                cand_rows["img_sim"] = img_sim_norm
                cand_rows["presence_boost"] = presence_boost_arr
                cand_rows["name_boost"] = name_boost_arr
                cand_rows["exact_name_boost"] = exact_name_boost_arr
                cand_rows["color_boost"] = color_boost_arr
                cand_rows["location_boost"] = location_boost_arr
                cand_rows["score"] = final_scores
                cand_rows["has_image"] = presence_flags > 0

                # optionally prefer items with images
                if only_with_images:
                    cand_rows = cand_rows[
                        cand_rows.apply(
                            lambda r: os.path.exists(str(r.get("image_path") or "")),
                            axis=1,
                        )
                    ]

                # sort: score first, then has_image if user prefers
                if show_images_first:
                    results = cand_rows.sort_values(
                        ["score", "has_image"], ascending=[False, False]
                    ).head(results_limit)
                else:
                    results = cand_rows.sort_values("score", ascending=False).head(results_limit)

                st.success(f"Found {len(results)} matches — results sorted by final fused score")
                for _, r in results.iterrows():
                    sc = float(r["score"])
                    # more permissive thresholds so good matches show as High
                    badge_label = "High" if sc >= 0.65 else ("Medium" if sc >= 0.45 else "Low")
                    contributed_by_image = (r["presence_boost"] > 0)
                    st.markdown("---")
                    card_cols = st.columns([3, 1])
                    with card_cols[0]:
                        st.markdown(f"**{r.get('item_name', '(no name)')}**")
                        if str(r.get("description", "")).strip():
                            st.write(f"_{r.get('description')}_")
                        small = []
                        if r.get("color", ""):
                            small.append(f"🎨 {r.get('color')}")
                        if r.get("location", ""):
                            small.append(f"📍 {r.get('location')}")
                        if small:
                            st.markdown(
                                "<span class='small'>" + " | ".join(small) + "</span>",
                                unsafe_allow_html=True,
                            )
                        # explain score
                        with st.expander("Why this score?"):
                            st.write(
                                {
                                    "cos_text": float(r.get("cos_text", 0)),
                                    "jaccard": float(r.get("jaccard", 0)),
                                    "text_prob": float(r.get("text_prob", 0)),
                                    "img_sim": float(r.get("img_sim", 0)),
                                    "presence_boost": float(r.get("presence_boost", 0)),
                                    "name_boost": float(r.get("name_boost", 0)),
                                    "exact_name_boost": float(r.get("exact_name_boost", 0)),
                                    "color_boost": float(r.get("color_boost", 0)),
                                    "location_boost": float(r.get("location_boost", 0)),
                                    "final_score": float(r.get("score", 0)),
                                }
                            )
                    with card_cols[1]:
                        imgp = resolve_local_display_image(r)
                        try:
                            st.image(
                                imgp,
                                caption=f"{sc:.3f} • {badge_label}"
                                + (" • image" if contributed_by_image else ""),
                                use_column_width=True,
                            )
                        except Exception:
                            st.write(f"{sc:.3f} • {badge_label}")
                        # feedback buttons
                        fid = (
                            r.get("item_id")
                            or r.get("id")
                            or r.get("uuid")
                            or uuid.uuid4().hex
                        )
                        if st.button("✓ Correct", key=f"c_{fid}", use_container_width=True):
                            rec = {
                                "lost_id": "user_input",
                                "found_id": str(fid),
                                "label": 1,
                                "ts": datetime.now().isoformat(),
                            }
                            try:
                                arr = json.load(open(FEEDBACK_JSON)) if os.path.exists(FEEDBACK_JSON) else []
                            except:
                                arr = []
                            arr.append(rec)
                            try:
                                json.dump(arr, open(FEEDBACK_JSON, "w"), indent=2)
                            except:
                                pass
                            st.success("Marked correct — thanks!")
                        if st.button("✗ Incorrect", key=f"i_{fid}", use_container_width=True):
                            rec = {
                                "lost_id": "user_input",
                                "found_id": str(fid),
                                "label": 0,
                                "ts": datetime.now().isoformat(),
                            }
                            try:
                                arr = json.load(open(FEEDBACK_JSON)) if os.path.exists(FEEDBACK_JSON) else []
                            except:
                                arr = []
                            arr.append(rec)
                            try:
                                json.dump(arr, open(FEEDBACK_JSON, "w"), indent=2)
                            except:
                                pass
                            st.info("Marked incorrect — thanks!")

# ---------------- Diagnostics Tab ----------------
with tabs[1]:
    st.header("Diagnostics")
    st.write("Rows:", df.shape[0])
    st.write(
        "Found items:",
        int((df.get("lost_or_found") == "found").sum())
        if "lost_or_found" in df.columns
        else "(unknown)",
    )
    st.write("Placeholders:", len(os.listdir(PLACEHOLDER_DIR)))
    if st.button("Show assigned image rows"):
        if "image_path" in df.columns:
            assigned = df[df["image_path"].astype(str).str.strip() != ""].copy()
            if assigned.empty:
                st.info("No rows have image_path set.")
            else:
                assigned["image_exists"] = assigned["image_path"].apply(
                    lambda p: os.path.exists(str(p))
                )
                st.dataframe(
                    assigned[["item_id", "item_name", "image_path", "image_exists"]].reset_index(
                        drop=True
                    )
                )
                st.write(
                    "Count with file present:",
                    int(assigned["image_exists"].sum()),
                    " / ",
                    assigned.shape[0],
                )
        else:
            st.info("No image_path column present in dataset.")
    if st.button("Regenerate placeholders (all)"):
        for _, rr in df.iterrows():
            _ = generate_placeholder(rr, force=True)
        st.success("Regenerated placeholders.")
    st.markdown("---")
    st.write("Embedder available:", embedder is not None)
    st.write("Precomputed embeddings present:", os.path.exists(IMAGE_EMB_NPZ))

# ---------------- Manage Tab ----------------
with tabs[2]:
    st.header("Manage")
    st.write("No drag-and-drop or file uploads in this build.")
    st.write(
        "To attach images to rows, set the image_path or image_url fields in your CSV "
        "and reload the dataset file in this folder."
    )
    if st.button("Show sample dataset (first 20)"):

        cols = [
            c
            for c in ["item_id", "item_name", "description", "color", "image_path"]
            if c in st.session_state.df.columns
        ]
        st.dataframe(st.session_state.df.head(20)[cols])

st.caption(
    "If images still show as missing: open Diagnostics → Show assigned image rows and paste the table output here. "
    "If any 'image_exists' are False, paste 2–3 example paths and your OS terminal output and I'll patch immediately."
)
