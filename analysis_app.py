"""
CoralSeg Annotation Analysis Dashboard
--------------------------------------
Standalone Streamlit app for inspecting the point annotations used to train
the patch-classifier / SegFormer model.

Run:
    streamlit run analysis_app.py --server.port 8502
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from utils import (
    load_labelset_from_json,
    load_annotations_from_df,
    normalize_image_name,
)

# ---------- defaults ----------
ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "input" / "annotations" / "annotations_confirmed.csv"
DEFAULT_IMG_DIR = ROOT / "input" / "images" / "Training Data"
LABELSET_PATH = ROOT / "input" / "labelset" / "labelset.json"

st.set_page_config(page_title="CoralSeg Annotation Analysis", layout="wide")

# Reuse the compact UI styling from webapp.py
st.markdown(
    """
<style>
    .stMarkdown p { margin-bottom: 0.3rem; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0.3rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }
    .stTabs [data-baseweb="tab"] { padding: 0.3rem 0.8rem; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("CoralSeg – Annotation Analysis")
st.caption(
    "Dashboard for the point annotations used to train the patch-classifier / SegFormer model."
)


# ---------- cached loaders ----------
@st.cache_data(show_spinner=False)
def load_labelset():
    with open(LABELSET_PATH, "r") as f:
        return load_labelset_from_json(json.load(f))


@st.cache_data(show_spinner="Loading annotations CSV…")
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV, normalise image names, keep useful columns."""
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalise label column (CSV uses 'Label code')
    if "Label" not in df.columns and "Label code" in df.columns:
        df = df.rename(columns={"Label code": "Label"})
    needed = ["Name", "Row", "Column", "Label"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df[needed].copy()
    df["NameNorm"] = df["Name"].astype(str).map(normalize_image_name)
    df["Label"] = df["Label"].astype(str)
    return df


@st.cache_data(show_spinner="Indexing image folder…")
def index_images(img_dir: str) -> dict:
    """Return mapping: normalized_name -> absolute path (recursive)."""
    p = Path(img_dir)
    if not p.exists():
        return {}
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    out = {}
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            out.setdefault(normalize_image_name(f.name), str(f))
    return out


@st.cache_data(show_spinner=False)
def build_class_table(df: pd.DataFrame, _labelset: list) -> pd.DataFrame:
    """Per-class point count, image count, plus labelset metadata."""
    by_label = df.groupby("Label")
    rows = []
    n_total_imgs = df["NameNorm"].nunique()
    meta_by_code = {e["Short Code"]: e for e in _labelset}
    for code, g in by_label:
        meta = meta_by_code.get(code, {})
        n_points = len(g)
        n_images = g["NameNorm"].nunique()
        rows.append(
            {
                "Short Code": code,
                "Name": meta.get("Name", code),
                "Functional Group": meta.get("Functional Group", "Unknown"),
                "Points": n_points,
                "Images": n_images,
                "% of points": 100 * n_points / len(df),
                "% of images": 100 * n_images / n_total_imgs,
                "Color": meta.get("Color Code", [128, 128, 128]),
            }
        )
    out = pd.DataFrame(rows).sort_values("Points", ascending=False).reset_index(drop=True)
    return out


def color_to_hex(c):
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
    return "#808080"


# ---------- sidebar ----------
st.sidebar.header("Data sources")
csv_path = st.sidebar.text_input("Annotations CSV", str(DEFAULT_CSV))
img_dir = st.sidebar.text_input("Image folder (recursive)", str(DEFAULT_IMG_DIR))

if not Path(csv_path).exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

labelset = load_labelset()
df = load_csv(csv_path)
img_index = index_images(img_dir)

color_by_code = {e["Short Code"]: color_to_hex(e.get("Color Code", [128, 128, 128])) for e in labelset}

# Top-level metrics
total_points = len(df)
total_images_csv = df["NameNorm"].nunique()
total_classes = df["Label"].nunique()
matched_imgs = sum(1 for n in df["NameNorm"].unique() if n in img_index)

c1, c2, c3, c4 = st.sidebar.columns(2) if False else (st.sidebar, None, None, None)
st.sidebar.markdown("---")
st.sidebar.metric("Total points", f"{total_points:,}")
st.sidebar.metric("Images in CSV", f"{total_images_csv:,}")
st.sidebar.metric("Images on disk (matched)", f"{matched_imgs:,} / {total_images_csv:,}")
st.sidebar.metric("Classes", total_classes)

class_table = build_class_table(df, labelset)

# ---------- tabs ----------
tab_img, tab_classes, tab_imbalance, tab_explorer = st.tabs(
    ["🖼️ Image viewer", "📋 Class overview", "⚖️ Class imbalance", "🔎 Class explorer"]
)

# ============== TAB 1: image viewer ==============
with tab_img:
    st.subheader("Inspect a single image")

    only_matched = st.checkbox("Only show images present on disk", value=True)
    names = sorted(df["NameNorm"].unique())
    if only_matched:
        names = [n for n in names if n in img_index]

    query = st.text_input(
        "Search filename", "", key="img_search",
        placeholder="Type a few characters to filter (e.g. 10b, G0258)…",
    )
    if query:
        q = query.lower()
        filtered = [n for n in names if q in n.lower()]
    else:
        filtered = names

    if not filtered:
        st.warning(
            "No images match." if query else "No images to show. Check the image folder path."
        )
    else:
        sel = st.selectbox(
            f"Image ({len(filtered):,} of {len(names):,} shown)", filtered
        )
        sub = df[df["NameNorm"] == sel]

        col_img, col_info = st.columns([3, 2])

        with col_img:
            path = img_index.get(sel)
            if path is None:
                st.warning("Image file not found on disk.")
            else:
                img = Image.open(path).convert("RGB")
                draw = ImageDraw.Draw(img)
                # Radius scales with image size
                r = max(4, min(img.size) // 200)
                for _, row in sub.iterrows():
                    x, y = int(row["Column"]), int(row["Row"])
                    fill = color_by_code.get(row["Label"], "#ff00ff")
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline="black")
                st.image(img, caption=f"{sel}  ({len(sub)} points)", use_container_width=True)

        with col_info:
            st.markdown(f"**Points in this image:** {len(sub)}")
            freq = (
                sub.groupby("Label")
                .size()
                .reset_index(name="Points")
                .sort_values("Points", ascending=False)
            )
            meta_by_code = {e["Short Code"]: e for e in labelset}
            freq["Name"] = freq["Label"].map(lambda c: meta_by_code.get(c, {}).get("Name", c))
            freq["Functional Group"] = freq["Label"].map(
                lambda c: meta_by_code.get(c, {}).get("Functional Group", "")
            )
            freq["Color"] = freq["Label"].map(lambda c: color_by_code.get(c, "#808080"))

            # Render a small HTML table with color swatches
            html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
            html.append(
                "<tr><th></th><th align='left'>Code</th><th align='left'>Name</th>"
                "<th align='left'>Group</th><th align='right'>Points</th></tr>"
            )
            for _, r_ in freq.iterrows():
                html.append(
                    f"<tr style='border-top:1px solid #eee;'>"
                    f"<td><span style='display:inline-block;width:14px;height:14px;"
                    f"background:{r_['Color']};border:1px solid #555;border-radius:3px;'></span></td>"
                    f"<td>{r_['Label']}</td><td>{r_['Name']}</td>"
                    f"<td>{r_['Functional Group']}</td><td align='right'>{r_['Points']}</td></tr>"
                )
            html.append("</table>")
            st.markdown("".join(html), unsafe_allow_html=True)

    # ---- Outlier detection ----
    st.markdown("---")
    st.subheader("Outlier images (point count out of range)")

    col_min, col_max = st.columns(2)
    min_pts = col_min.number_input(
        "Minimum points (inclusive)", min_value=0, value=50, step=10,
        help="Images with fewer than this many points are flagged as outliers.",
    )
    max_pts = col_max.number_input(
        "Maximum points (inclusive)", min_value=0, value=150, step=10,
        help="Images with more than this many points are flagged as outliers.",
    )

    exclude_str = st.text_input(
        "Exclude images whose filename contains (case-insensitive)",
        value="trainingMM",
        help="Any image whose normalised name contains this substring will be removed from the cleaned CSV.",
    )

    pts_per_img = df.groupby("NameNorm").size().reset_index(name="Points")
    outliers = pts_per_img[
        (pts_per_img["Points"] < min_pts) | (pts_per_img["Points"] > max_pts)
    ].sort_values("Points", ascending=False).reset_index(drop=True)

    # Images excluded by the name-substring filter
    exclude_lower = exclude_str.strip().lower()
    if exclude_lower:
        string_excluded = [n for n in df["NameNorm"].unique() if exclude_lower in n.lower()]
    else:
        string_excluded = []

    if outliers.empty and not string_excluded:
        st.success(f"No outliers found (range: {min_pts}–{max_pts} points, inclusive) and no name-match exclusions.")
    else:
        if not outliers.empty:
            st.warning(f"{len(outliers)} image(s) outside the {min_pts}–{max_pts} range.")
            st.dataframe(outliers.rename(columns={"NameNorm": "Image"}), use_container_width=True, hide_index=True)
        if string_excluded:
            st.warning(f"{len(string_excluded)} image(s) excluded by name filter '{exclude_str}'.")

        # Build cleaned dataframe: remove both outlier and string-excluded images
        all_excluded = set(outliers["NameNorm"]) | set(string_excluded)
        clean_df = df[~df["NameNorm"].isin(all_excluded)].drop(columns=["NameNorm"])
        if "Label code" not in clean_df.columns and "Label" in clean_df.columns:
            clean_df = clean_df.rename(columns={"Label": "Label code"})

        CLEANED_PATH = ROOT / "input" / "annotations" / "annotations_cleaned.csv"

        if st.button("💾 Save cleaned CSV to disk"):
            try:
                clean_df.to_csv(CLEANED_PATH, index=False)
                st.success(
                    f"✅ Saved **{len(clean_df):,}** rows to `{CLEANED_PATH}`  \n"
                    f"Removed: **{len(outliers)}** point-count outlier(s), "
                    f"**{len(string_excluded)}** name-filter exclusion(s)."
                )
            except Exception as e:
                st.error(f"❌ Could not save file: {e}")

        st.download_button(
            "⬇️ Download list of excluded filenames",
            "\n".join(sorted(all_excluded)).encode("utf-8"),
            "excluded_images.txt",
            "text/plain",
        )


# ============== TAB 2: class overview ==============
with tab_classes:
    st.subheader("Per-class statistics")
    show = class_table.copy()
    show["Color"] = show["Color"].map(color_to_hex)

    # Custom HTML table so we can render the colour swatch
    html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
    html.append(
        "<tr style='text-align:left;border-bottom:2px solid #999;'>"
        "<th></th><th>Code</th><th>Name</th><th>Group</th>"
        "<th align='right'>Points</th><th align='right'>Images</th>"
        "<th align='right'>% pts</th><th align='right'>% imgs</th></tr>"
    )
    for _, r_ in show.iterrows():
        html.append(
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td><span style='display:inline-block;width:14px;height:14px;"
            f"background:{r_['Color']};border:1px solid #555;border-radius:3px;'></span></td>"
            f"<td>{r_['Short Code']}</td><td>{r_['Name']}</td>"
            f"<td>{r_['Functional Group']}</td>"
            f"<td align='right'>{r_['Points']:,}</td>"
            f"<td align='right'>{r_['Images']:,}</td>"
            f"<td align='right'>{r_['% of points']:.2f}</td>"
            f"<td align='right'>{r_['% of images']:.2f}</td></tr>"
        )
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    csv_bytes = class_table.drop(columns=["Color"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download class table (CSV)", csv_bytes, "class_table.csv", "text/csv")


# ============== TAB 3: class imbalance ==============
with tab_imbalance:
    st.subheader("Class imbalance overview")

    metric = st.radio(
        "Metric",
        ["Points", "Images"],
        horizontal=True,
        help="Points = number of point annotations per class. "
        "Images = number of distinct images that contain at least one point of that class.",
    )

    order = class_table.sort_values(metric, ascending=False).reset_index(drop=True)
    vals = order[metric].astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", len(order))
    c2.metric(f"Total {metric.lower()}", f"{vals.sum():,}")
    c3.metric("Median per class", f"{int(vals.median()):,}")
    c4.metric(
        "Max / Min ratio",
        f"{(vals.max() / max(vals.min(), 1)):.0f}×",
        help="Largest class divided by smallest class (rough imbalance indicator).",
    )

    # Bar chart: most_common -> least_common
    chart_df = order[["Short Code", metric]].set_index("Short Code")
    st.bar_chart(chart_df, height=420)

    with st.expander("Table sorted by selected metric"):
        st.dataframe(
            order[
                ["Short Code", "Name", "Functional Group", "Points", "Images", "% of points", "% of images"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============== TAB 4: class explorer ==============
with tab_explorer:
    st.subheader("Images containing a specific class")

    all_options = class_table["Short Code"].tolist()
    all_labels = [
        f"{c}  –  {class_table.loc[class_table['Short Code'] == c, 'Name'].iloc[0]}  "
        f"({int(class_table.loc[class_table['Short Code'] == c, 'Images'].iloc[0])} imgs)"
        for c in all_options
    ]

    query = st.text_input(
        "Search class", "", key="class_search",
        placeholder="Type to filter by code, name or group (e.g. unk, algae, SG)…",
    )
    if query:
        q = query.lower()
        # Match against the short code, full name and functional group
        pairs = [
            (code, lbl)
            for code, lbl in zip(all_options, all_labels)
            if q in lbl.lower()
            or q
            in class_table.loc[class_table["Short Code"] == code, "Functional Group"]
            .iloc[0]
            .lower()
        ]
    else:
        pairs = list(zip(all_options, all_labels))

    if not pairs:
        st.warning("No class matches your search.")
        st.stop()

    options = [c for c, _ in pairs]
    labels_for_select = [l for _, l in pairs]
    idx = st.selectbox(
        f"Class ({len(options)} of {len(all_options)} shown)",
        range(len(options)),
        format_func=lambda i: labels_for_select[i],
    )
    sel_code = options[idx]

    sub = df[df["Label"] == sel_code]
    img_counts = (
        sub.groupby("NameNorm").size().reset_index(name="Points of this class").sort_values(
            "Points of this class", ascending=False
        )
    )
    img_counts["On disk"] = img_counts["NameNorm"].map(lambda n: "✓" if n in img_index else "—")

    st.markdown(
        f"**{sel_code}** appears in **{len(img_counts):,}** images "
        f"with **{len(sub):,}** total points."
    )

    st.dataframe(
        img_counts.rename(columns={"NameNorm": "Image"}),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    csv_bytes = img_counts.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download image list for {sel_code}",
        csv_bytes,
        f"images_{sel_code}.csv",
        "text/csv",
    )
