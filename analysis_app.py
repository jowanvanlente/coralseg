"""CoralSeg Annotation Analysis Dashboard
--------------------------------------
Combined analysis of CoralNet point annotations (CSV) and Roboflow
polygon annotations (COCO JSON) for training-data class-imbalance inspection.

Run:
    streamlit run analysis_app.py --server.port 8502
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from utils import (
    load_labelset_from_json,
    load_coco_annotations,
    normalize_image_name,
    _demangle_roboflow_name,
    _ROBOFLOW_NAME_TO_SHORT_CODE,
    LABEL_META_BY_SHORT_CODE,
)

# ---------- defaults ----------
ROOT = Path(__file__).parent
DEFAULT_CSV = ROOT / "input" / "annotations" / "annotations_confirmed_merged.csv"
DEFAULT_COCO = ROOT / "input" / "annotations" / "Roboflow" / "20260508_annotations_merged.coco.json"
DEFAULT_IMG_DIR = ROOT / "input" / "images" / "Training Data"
LABELSET_PATH = ROOT / "input" / "labelset" / "labelset_merged.json"

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
    "Combined dashboard: CoralNet point annotations + Roboflow polygon annotations."
)


# ---------- cached loaders ----------
@st.cache_data(show_spinner=False)
def load_labelset():
    with open(LABELSET_PATH, "r") as f:
        return load_labelset_from_json(json.load(f))


@st.cache_data(show_spinner="Loading CoralNet CSV…")
def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV, normalise image names, keep useful columns."""
    df = pd.read_csv(csv_path, low_memory=False)
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


@st.cache_data(show_spinner="Loading Roboflow COCO…")
def load_coco(coco_path: str) -> pd.DataFrame:
    """Load COCO JSON into a flat DataFrame with NameNorm, Label, Area."""
    return load_coco_annotations(coco_path)


@st.cache_data(show_spinner="Loading raw COCO JSON…")
def load_coco_raw(coco_path: str) -> dict:
    """Load the raw COCO JSON dict for validation inspection."""
    with open(coco_path, "r") as f:
        return json.load(f)


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
def build_class_table(
    df_coralnet: pd.DataFrame,
    df_roboflow: pd.DataFrame,
    _labelset: list,
) -> pd.DataFrame:
    """Per-class stats combining CoralNet points and Roboflow polygons."""
    meta_by_code = {e["Short Code"]: e for e in _labelset}

    # CoralNet aggregation
    cn_by_label = df_coralnet.groupby("Label")
    cn_stats = {}
    for code, g in cn_by_label:
        cn_stats[code] = {"cn_points": len(g), "cn_images": g["NameNorm"].nunique()}

    # Roboflow aggregation
    rf_stats = {}
    if len(df_roboflow):
        rf_by_label = df_roboflow.groupby("Label")
        for code, g in rf_by_label:
            rf_stats[code] = {
                "rf_instances": len(g),
                "rf_images": g["NameNorm"].nunique(),
                "rf_area": int(g["Area"].sum()),
            }

    all_codes = sorted(set(cn_stats) | set(rf_stats))
    total_cn = len(df_coralnet)
    total_rf = len(df_roboflow)
    rows = []
    for code in all_codes:
        meta = meta_by_code.get(code, {})
        cn = cn_stats.get(code, {"cn_points": 0, "cn_images": 0})
        rf = rf_stats.get(code, {"rf_instances": 0, "rf_images": 0, "rf_area": 0})
        rows.append({
            "Short Code": code,
            "Name": meta.get("Name", code),
            "Functional Group": meta.get("Functional Group", "Unknown"),
            "CN Points": cn["cn_points"],
            "CN Images": cn["cn_images"],
            "RF Instances": rf["rf_instances"],
            "RF Images": rf["rf_images"],
            "RF Area (px)": rf["rf_area"],
            "% CN pts": 100 * cn["cn_points"] / max(total_cn, 1),
            "% RF inst": 100 * rf["rf_instances"] / max(total_rf, 1),
            "Color": meta.get("Color Code", [128, 128, 128]),
        })
    out = pd.DataFrame(rows).sort_values("CN Points", ascending=False).reset_index(drop=True)
    return out


def color_to_hex(c):
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
    return "#808080"


# ---------- sidebar ----------
st.sidebar.header("Data sources")
csv_path = st.sidebar.text_input("CoralNet CSV", str(DEFAULT_CSV))
coco_path = st.sidebar.text_input("Roboflow COCO JSON", str(DEFAULT_COCO))
img_dir = st.sidebar.text_input("Image folder (recursive)", str(DEFAULT_IMG_DIR))

if not Path(csv_path).exists():
    st.error(f"CSV not found: {csv_path}")
    st.stop()

labelset = load_labelset()
df = load_csv(csv_path)
img_index = index_images(img_dir)

# Load Roboflow COCO (empty DF if file missing)
if Path(coco_path).exists():
    df_rf = load_coco(coco_path)
else:
    st.sidebar.warning("COCO JSON not found – showing CoralNet only.")
    df_rf = pd.DataFrame(columns=["NameNorm", "Label", "Area", "ImgW", "ImgH"])

# Build raw COCO index for polygon rendering (shared by image viewer and validation tab)
coco_raw = None
coco_cat_map: dict = {}
coco_img_meta: dict = {}
coco_anns_by_img: dict = {}
if Path(coco_path).exists():
    coco_raw = load_coco_raw(coco_path)
    coco_cat_map = {c["id"]: c["name"] for c in coco_raw["categories"]}
    for im in coco_raw["images"]:
        extra_name = im.get("extra", {}).get("name", im["file_name"])
        demangled = _demangle_roboflow_name(extra_name)
        norm = normalize_image_name(demangled)
        coco_img_meta[im["id"]] = {
            "file_name": im["file_name"],
            "extra_name": extra_name,
            "demangled": demangled,
            "norm": norm,
            "width": im["width"],
            "height": im["height"],
            "tags": im.get("extra", {}).get("user_tags", []),
        }
    for a in coco_raw["annotations"]:
        iid = a["image_id"]
        if iid not in coco_img_meta:
            continue
        norm = coco_img_meta[iid]["norm"]
        coco_anns_by_img.setdefault(norm, []).append(a)

color_by_code = {e["Short Code"]: color_to_hex(e.get("Color Code", [128, 128, 128])) for e in labelset}

# Top-level metrics
total_cn_points = len(df)
total_cn_images = df["NameNorm"].nunique()
total_rf_instances = len(df_rf)
total_rf_images = df_rf["NameNorm"].nunique() if len(df_rf) else 0
all_images = set(df["NameNorm"].unique())
if len(df_rf):
    all_images |= set(df_rf["NameNorm"].unique())
overlap_images = set(df["NameNorm"].unique()) & set(df_rf["NameNorm"].unique()) if len(df_rf) else set()
total_classes = len(set(df["Label"].unique()) | (set(df_rf["Label"].unique()) if len(df_rf) else set()))
matched_imgs = sum(1 for n in all_images if n in img_index)

st.sidebar.markdown("---")
st.sidebar.markdown("**CoralNet (point annotations)**")
st.sidebar.metric("Points", f"{total_cn_points:,}")
st.sidebar.metric("Images", f"{total_cn_images:,}")
st.sidebar.markdown("**Roboflow (polygon annotations)**")
st.sidebar.metric("Polygon instances", f"{total_rf_instances:,}")
st.sidebar.metric("Images", f"{total_rf_images:,}")
st.sidebar.markdown("---")
st.sidebar.metric("Unique images (union)", f"{len(all_images):,}")
st.sidebar.metric("Images in both sources", f"{len(overlap_images):,}")
st.sidebar.metric("Images on disk", f"{matched_imgs:,}")
st.sidebar.metric("Total classes", total_classes)

class_table = build_class_table(df, df_rf, labelset)

# ---------- tabs ----------
tab_img, tab_classes, tab_imbalance, tab_explorer, tab_validate = st.tabs(
    ["🖼️ Image viewer", "📋 Class overview", "⚖️ Class imbalance",
     "🔎 Class explorer", "🔬 Data validation"]
)

# ============== TAB 1: image viewer ==============
with tab_img:
    st.subheader("Inspect a single image")

    rf_images_set = set(df_rf["NameNorm"].unique()) if len(df_rf) else set()
    col_f1, col_f2 = st.columns(2)
    only_matched = col_f1.checkbox("Only show images present on disk", value=True)
    only_has_roboflow = col_f2.checkbox(
        "Only show images with Roboflow annotations",
        value=False,
        disabled=len(rf_images_set) == 0,
    )
    # Union of images from both sources
    all_img_names = sorted(set(df["NameNorm"].unique()) | rf_images_set)
    if only_matched:
        all_img_names = [n for n in all_img_names if n in img_index]
    if only_has_roboflow:
        all_img_names = [n for n in all_img_names if n in rf_images_set]

    query = st.text_input(
        "Search filename", "", key="img_search",
        placeholder="Type a few characters to filter (e.g. 10b, G0258)…",
    )
    if query:
        q = query.lower()
        filtered = [n for n in all_img_names if q in n.lower()]
    else:
        filtered = all_img_names

    if not filtered:
        st.warning(
            "No images match." if query else "No images to show. Check the image folder path."
        )
    else:
        sel = st.selectbox(
            f"Image ({len(filtered):,} of {len(all_img_names):,} shown)", filtered
        )
        sub_cn = df[df["NameNorm"] == sel]
        sub_rf = df_rf[df_rf["NameNorm"] == sel] if len(df_rf) else df_rf.iloc[0:0]

        col_img, col_info = st.columns([3, 2])

        with col_img:
            path = img_index.get(sel)
            if path is None:
                st.warning("Image file not found on disk.")
            else:
                img = Image.open(path).convert("RGBA")
                # Draw Roboflow polygon overlays first (rendered behind CoralNet points)
                rf_anns_for_img = coco_anns_by_img.get(sel, [])
                if rf_anns_for_img:
                    try:
                        from pycocotools import mask as coco_mask_util
                        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        draw_fill = ImageDraw.Draw(overlay)
                        for ann in rf_anns_for_img:
                            cat_name = coco_cat_map.get(ann["category_id"], "")
                            sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cat_name) or cat_name
                            meta_entry = next((e for e in labelset if e["Short Code"] == sc), None)
                            if meta_entry and "Color Code" in meta_entry:
                                cc = meta_entry["Color Code"]
                                rgb = (int(cc[0]), int(cc[1]), int(cc[2]))
                            else:
                                rgb = (255, 0, 255)
                            fill_rgba = rgb + (60,)
                            outline_rgba = rgb + (230,)
                            seg = ann.get("segmentation")
                            if isinstance(seg, list):
                                for ring in seg:
                                    if len(ring) >= 6:
                                        pts = [(ring[i], ring[i + 1]) for i in range(0, len(ring), 2)]
                                        draw_fill.polygon(pts, fill=fill_rgba)
                                        draw_fill.line(pts + [pts[0]], fill=outline_rgba, width=3)
                            elif isinstance(seg, dict):
                                try:
                                    if isinstance(seg["counts"], list):
                                        rle = coco_mask_util.frPyObjects([seg], seg["size"][0], seg["size"][1])[0]
                                    else:
                                        rle = seg
                                    mask_arr = coco_mask_util.decode(rle)
                                    mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                                    color_layer = Image.new("RGBA", img.size, fill_rgba)
                                    overlay.paste(color_layer, mask=mask_img)
                                except Exception:
                                    bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
                                    draw_fill.rectangle([bx, by, bx + bw, by + bh], fill=fill_rgba, outline=outline_rgba)
                        img = Image.alpha_composite(img, overlay)
                    except ImportError:
                        pass  # pycocotools not available
                img = img.convert("RGB")
                # Draw CoralNet points on top
                draw = ImageDraw.Draw(img)
                r = max(4, min(img.size) // 200)
                for _, row in sub_cn.iterrows():
                    x, y = int(row["Column"]), int(row["Row"])
                    fill = color_by_code.get(row["Label"], "#ff00ff")
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline="black")
                caption_parts = []
                if len(sub_cn):
                    caption_parts.append(f"{len(sub_cn)} CoralNet points")
                if len(sub_rf):
                    caption_parts.append(f"{len(sub_rf)} Roboflow polygons")
                st.image(img, caption=f"{sel}  ({', '.join(caption_parts) or 'no annotations'})", use_container_width=True)

        with col_info:
            meta_by_code = {e["Short Code"]: e for e in labelset}

            # CoralNet annotations table
            if len(sub_cn):
                st.markdown(f"**CoralNet points:** {len(sub_cn)}")
                freq = (
                    sub_cn.groupby("Label").size().reset_index(name="Points")
                    .sort_values("Points", ascending=False)
                )
                freq["Name"] = freq["Label"].map(lambda c: meta_by_code.get(c, {}).get("Name", c))
                freq["Color"] = freq["Label"].map(lambda c: color_by_code.get(c, "#808080"))
                html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
                html.append("<tr><th></th><th align='left'>Code</th><th align='left'>Name</th><th align='right'>Pts</th></tr>")
                for _, r_ in freq.iterrows():
                    html.append(
                        f"<tr style='border-top:1px solid #eee;'>"
                        f"<td><span style='display:inline-block;width:14px;height:14px;"
                        f"background:{r_['Color']};border:1px solid #555;border-radius:3px;'></span></td>"
                        f"<td>{r_['Label']}</td><td>{r_['Name']}</td><td align='right'>{r_['Points']}</td></tr>"
                    )
                html.append("</table>")
                st.markdown("".join(html), unsafe_allow_html=True)
            else:
                st.info("No CoralNet annotations for this image.")

            # Roboflow annotations table
            if len(sub_rf):
                st.markdown(f"**Roboflow polygons:** {len(sub_rf)}")
                rf_freq = (
                    sub_rf.groupby("Label").agg(Instances=("Label", "size"), Area=("Area", "sum"))
                    .reset_index().sort_values("Area", ascending=False)
                )
                rf_freq["Name"] = rf_freq["Label"].map(lambda c: meta_by_code.get(c, {}).get("Name", c))
                rf_freq["Color"] = rf_freq["Label"].map(lambda c: color_by_code.get(c, "#808080"))
                html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
                html.append("<tr><th></th><th align='left'>Code</th><th align='left'>Name</th>"
                            "<th align='right'>Inst</th><th align='right'>Area (px)</th></tr>")
                for _, r_ in rf_freq.iterrows():
                    html.append(
                        f"<tr style='border-top:1px solid #eee;'>"
                        f"<td><span style='display:inline-block;width:14px;height:14px;"
                        f"background:{r_['Color']};border:1px solid #555;border-radius:3px;'></span></td>"
                        f"<td>{r_['Label']}</td><td>{r_['Name']}</td>"
                        f"<td align='right'>{int(r_['Instances'])}</td>"
                        f"<td align='right'>{int(r_['Area']):,}</td></tr>"
                    )
                html.append("</table>")
                st.markdown("".join(html), unsafe_allow_html=True)
            else:
                st.info("No Roboflow annotations for this image.")

    # ---- Outlier detection (CoralNet only) ----
    st.markdown("---")
    st.subheader("Outlier images (CoralNet point count)")

    use_range = st.checkbox("Filter by point count range", value=True)
    col_min, col_max = st.columns(2)
    min_pts = col_min.number_input(
        "Minimum points (inclusive)", min_value=0, value=50, step=10,
        help="Images with fewer than this many points are flagged as outliers.",
        disabled=not use_range,
    )
    max_pts = col_max.number_input(
        "Maximum points (inclusive)", min_value=0, value=150, step=10,
        help="Images with more than this many points are flagged as outliers.",
        disabled=not use_range,
    )

    use_name = st.checkbox("Filter by filename", value=True)
    exclude_str = st.text_input(
        "Exclude images whose filename contains (case-insensitive)",
        value="trainingMM",
        help="Any image whose normalised name contains this substring will be removed from the cleaned CSV.",
        disabled=not use_name,
    )

    pts_per_img = df.groupby("NameNorm").size().reset_index(name="Points")
    if use_range:
        outliers = pts_per_img[
            (pts_per_img["Points"] < min_pts) | (pts_per_img["Points"] > max_pts)
        ].sort_values("Points", ascending=False).reset_index(drop=True)
    else:
        outliers = pts_per_img.iloc[0:0]

    exclude_lower = exclude_str.strip().lower()
    if use_name and exclude_lower:
        string_excluded = [n for n in df["NameNorm"].unique() if exclude_lower in n.lower()]
    else:
        string_excluded = []

    if outliers.empty and not string_excluded:
        st.success(f"No outliers (range {min_pts}–{max_pts}) and no name-match exclusions.")
    else:
        if not outliers.empty:
            st.warning(f"{len(outliers)} image(s) outside the {min_pts}–{max_pts} range.")
            st.dataframe(outliers.rename(columns={"NameNorm": "Image"}), width="stretch", hide_index=True)
        if string_excluded:
            st.warning(f"{len(string_excluded)} image(s) excluded by name filter '{exclude_str}'.")

        all_excluded = set(outliers["NameNorm"]) | set(string_excluded)
        st.info(f"**Total excluded:** {len(all_excluded)} images  →  **{total_cn_images - len(all_excluded)}** images remain")
        clean_df = df[~df["NameNorm"].isin(all_excluded)].drop(columns=["NameNorm"])
        if "Label code" not in clean_df.columns and "Label" in clean_df.columns:
            clean_df = clean_df.rename(columns={"Label": "Label code"})

        CLEANED_PATH = ROOT / "input" / "annotations" / "annotations_cleaned.csv"

        if st.button("💾 Save cleaned CSV to disk"):
            try:
                clean_df.to_csv(CLEANED_PATH, index=False)
                st.success(
                    f"Saved **{len(clean_df):,}** rows to `{CLEANED_PATH}`  \n"
                    f"Removed: **{len(outliers)}** point-count outlier(s), "
                    f"**{len(string_excluded)}** name-filter exclusion(s)."
                )
            except Exception as e:
                st.error(f"Could not save file: {e}")

        st.download_button(
            "⬇️ Download list of excluded filenames",
            "\n".join(sorted(all_excluded)).encode("utf-8"),
            "excluded_images.txt",
            "text/plain",
        )


# ============== TAB 2: class overview ==============
with tab_classes:
    st.subheader("Per-class statistics (combined)")
    show = class_table.copy()
    show["Color"] = show["Color"].map(color_to_hex)

    html = ["<table style='width:100%;font-size:0.82rem;border-collapse:collapse;'>"]
    html.append(
        "<tr style='text-align:left;border-bottom:2px solid #999;'>"
        "<th></th><th>Code</th><th>Name</th><th>Group</th>"
        "<th align='right'>CN Pts</th><th align='right'>CN Imgs</th>"
        "<th align='right'>RF Inst</th><th align='right'>RF Imgs</th>"
        "<th align='right'>RF Area</th>"
        "<th align='right'>% CN</th><th align='right'>% RF</th></tr>"
    )
    for _, r_ in show.iterrows():
        html.append(
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td><span style='display:inline-block;width:14px;height:14px;"
            f"background:{r_['Color']};border:1px solid #555;border-radius:3px;'></span></td>"
            f"<td>{r_['Short Code']}</td><td>{r_['Name']}</td>"
            f"<td>{r_['Functional Group']}</td>"
            f"<td align='right'>{int(r_['CN Points']):,}</td>"
            f"<td align='right'>{int(r_['CN Images']):,}</td>"
            f"<td align='right'>{int(r_['RF Instances']):,}</td>"
            f"<td align='right'>{int(r_['RF Images']):,}</td>"
            f"<td align='right'>{int(r_['RF Area (px)']):,}</td>"
            f"<td align='right'>{r_['% CN pts']:.2f}</td>"
            f"<td align='right'>{r_['% RF inst']:.2f}</td></tr>"
        )
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.caption("CN = CoralNet point annotations · RF = Roboflow polygon annotations · Area in pixels²")

    csv_bytes = class_table.drop(columns=["Color"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download class table (CSV)", csv_bytes, "class_table.csv", "text/csv")

    # Classes only in one source
    cn_only = class_table[(class_table["CN Points"] > 0) & (class_table["RF Instances"] == 0)]
    rf_only = class_table[(class_table["CN Points"] == 0) & (class_table["RF Instances"] > 0)]
    if len(cn_only) or len(rf_only):
        with st.expander("Classes present in only one source"):
            if len(cn_only):
                st.markdown(f"**CoralNet only** ({len(cn_only)} classes): "
                            + ", ".join(cn_only["Short Code"].tolist()))
            if len(rf_only):
                st.markdown(f"**Roboflow only** ({len(rf_only)} classes): "
                            + ", ".join(rf_only["Short Code"].tolist()))


# ============== TAB 3: class imbalance ==============
with tab_imbalance:
    st.subheader("Class imbalance overview")

    metric = st.radio(
        "Metric",
        ["CN Points", "RF Instances", "RF Area (px)", "CN Images", "RF Images"],
        horizontal=True,
        help="CN Points = CoralNet point count · RF Instances = Roboflow polygon count · "
             "RF Area = total polygon area in px² · Images = distinct images with at least one annotation.",
    )

    order = class_table[class_table[metric] > 0].sort_values(metric, ascending=False).reset_index(drop=True)
    vals = order[metric].astype(float)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", len(order))
    c2.metric(f"Total", f"{int(vals.sum()):,}")
    c3.metric("Median", f"{int(vals.median()):,}" if len(vals) else "—")
    c4.metric(
        "Max / Min",
        f"{(vals.max() / max(vals.min(), 1)):.0f}×" if len(vals) else "—",
        help="Largest / smallest class (rough imbalance indicator).",
    )

    chart_df = order[["Short Code", metric]].reset_index(drop=True)
    st.bar_chart(chart_df, x="Short Code", y=metric, height=420)

    # Side-by-side comparison: CoralNet vs Roboflow distribution
    if len(df_rf):
        st.markdown("---")
        st.markdown("**Distribution comparison (% of total per source)**")
        compare = class_table[["Short Code", "% CN pts", "% RF inst"]].copy()
        compare = compare[(compare["% CN pts"] > 0) | (compare["% RF inst"] > 0)]
        compare = compare.sort_values("% CN pts", ascending=False).reset_index(drop=True)
        compare = compare.rename(columns={"% CN pts": "CoralNet %", "% RF inst": "Roboflow %"})
        st.bar_chart(compare, x="Short Code", y=["CoralNet %", "Roboflow %"], height=420)

        st.caption("This chart shows whether Roboflow annotations reinforce or counteract "
                   "the CoralNet class imbalance. Classes where the blue bar exceeds orange "
                   "are over-represented in CoralNet relative to Roboflow.")

    with st.expander("Table sorted by selected metric"):
        st.dataframe(
            order[["Short Code", "Name", "Functional Group",
                   "CN Points", "CN Images", "RF Instances", "RF Images",
                   "RF Area (px)", "% CN pts", "% RF inst"]],
            width="stretch",
            hide_index=True,
        )


# ============== TAB 4: class explorer ==============
with tab_explorer:
    st.subheader("Images containing a specific class")

    all_options = class_table["Short Code"].tolist()
    all_labels = [
        f"{c}  –  {class_table.loc[class_table['Short Code'] == c, 'Name'].iloc[0]}  "
        f"(CN:{int(class_table.loc[class_table['Short Code'] == c, 'CN Points'].iloc[0])}"
        f" RF:{int(class_table.loc[class_table['Short Code'] == c, 'RF Instances'].iloc[0])})"
        for c in all_options
    ]

    query = st.text_input(
        "Search class", "", key="class_search",
        placeholder="Type to filter by code, name or group (e.g. unk, algae, SG)…",
    )
    if query:
        q = query.lower()
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

    # CoralNet images
    sub_cn = df[df["Label"] == sel_code]
    cn_img_counts = (
        sub_cn.groupby("NameNorm").size().reset_index(name="CN Points")
        .sort_values("CN Points", ascending=False)
    )

    # Roboflow images
    sub_rf = df_rf[df_rf["Label"] == sel_code] if len(df_rf) else df_rf.iloc[0:0]
    rf_img_counts = (
        sub_rf.groupby("NameNorm").agg(
            **{"RF Instances": ("Label", "size"), "RF Area": ("Area", "sum")}
        ).reset_index().sort_values("RF Instances", ascending=False)
    ) if len(sub_rf) else pd.DataFrame(columns=["NameNorm", "RF Instances", "RF Area"])

    # Merge both
    merged = cn_img_counts.merge(rf_img_counts, on="NameNorm", how="outer")
    merged = merged.infer_objects(copy=False).fillna(0)
    merged["CN Points"] = merged["CN Points"].astype(int)
    merged["RF Instances"] = merged["RF Instances"].astype(int)
    merged["RF Area"] = merged["RF Area"].astype(int)
    merged["On disk"] = merged["NameNorm"].map(lambda n: "✓" if n in img_index else "—")
    merged = merged.sort_values("CN Points", ascending=False).reset_index(drop=True)

    total_cn = int(merged["CN Points"].sum())
    total_rf = int(merged["RF Instances"].sum())
    st.markdown(
        f"**{sel_code}** — {len(merged):,} images · "
        f"{total_cn:,} CoralNet points · {total_rf:,} Roboflow polygons"
    )

    st.dataframe(
        merged.rename(columns={"NameNorm": "Image"}),
        width="stretch",
        hide_index=True,
        height=500,
    )

    csv_bytes = merged.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download image list for {sel_code}",
        csv_bytes,
        f"images_{sel_code}.csv",
        "text/csv",
    )


# ============== TAB 5: data validation ==============
with tab_validate:
    st.subheader("Data Validation — CoralNet CSV vs Roboflow COCO")
    st.caption("Critical audit of image matching, class mapping, and annotation integrity.")

    coco_available = Path(coco_path).exists()
    if not coco_available:
        st.error("Roboflow COCO JSON not found — cannot run validation.")
        st.stop()

    csv_name_set = set(df["NameNorm"].unique())
    csv_label_set = set(df["Label"].unique())

    # ---- Section 0: Roboflow Polygon Viewer ----
    st.markdown("---")
    st.markdown("### 0. Roboflow Polygon Viewer")
    st.markdown("Browse Roboflow-annotated images with polygon regions drawn. "
                "Each class is shown in its own colour.")

    # Image picker (all Roboflow images)
    rf_img_norms = sorted(set(info["norm"] for info in coco_img_meta.values()))

    rf_search = st.text_input(
        "Search Roboflow image", "", key="rf_img_search",
        placeholder="Filter by filename…",
    )
    if rf_search:
        rf_filtered = [n for n in rf_img_norms if rf_search.lower() in n.lower()]
    else:
        rf_filtered = rf_img_norms

    if not rf_filtered:
        st.warning("No images match your filter.")
    else:
        rf_sel = st.selectbox(
            f"Roboflow image ({len(rf_filtered):,} of {len(rf_img_norms):,})",
            rf_filtered, key="rf_img_sel",
        )

        # Get annotations for this image
        rf_anns = coco_anns_by_img.get(rf_sel, [])

        col_rf_img, col_rf_info = st.columns([3, 2])

        with col_rf_img:
            path = img_index.get(rf_sel)
            if path is None:
                st.warning("Image file not found on disk — cannot render polygons.")
            else:
                from pycocotools import mask as coco_mask_util

                img = Image.open(path).convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                draw_overlay = ImageDraw.Draw(overlay)
                draw_outline = ImageDraw.Draw(img.convert("RGBA"))

                # Re-compose: base + semi-transparent fills + outlines
                base = img.copy()
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw_fill = ImageDraw.Draw(overlay)

                for ann in rf_anns:
                    cat_name = coco_cat_map.get(ann["category_id"], "")
                    sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cat_name) or cat_name
                    # Get color
                    meta_entry = next((e for e in labelset if e["Short Code"] == sc), None)
                    if meta_entry and "Color Code" in meta_entry:
                        c = meta_entry["Color Code"]
                        rgb = (int(c[0]), int(c[1]), int(c[2]))
                    else:
                        rgb = (255, 0, 255)
                    fill_rgba = rgb + (60,)
                    outline_rgba = rgb + (220,)

                    seg = ann.get("segmentation")
                    if isinstance(seg, list):
                        # Polygon format: list of [x1,y1,x2,y2,...] rings
                        for ring in seg:
                            if len(ring) >= 6:
                                pts = [(ring[i], ring[i+1]) for i in range(0, len(ring), 2)]
                                draw_fill.polygon(pts, fill=fill_rgba)
                                draw_fill.polygon(pts, outline=outline_rgba)
                    elif isinstance(seg, dict):
                        # RLE format — decode to binary mask
                        try:
                            if isinstance(seg["counts"], list):
                                rle = coco_mask_util.frPyObjects([seg], seg["size"][0], seg["size"][1])[0]
                            else:
                                rle = seg
                            mask_arr = coco_mask_util.decode(rle)
                            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                            color_layer = Image.new("RGBA", base.size, fill_rgba)
                            overlay.paste(color_layer, mask=mask_img)
                        except Exception:
                            # Fallback: draw bounding box
                            bx, by, bw, bh = ann.get("bbox", [0, 0, 0, 0])
                            draw_fill.rectangle(
                                [bx, by, bx + bw, by + bh],
                                fill=fill_rgba, outline=outline_rgba
                            )

                result = Image.alpha_composite(base, overlay).convert("RGB")
                st.image(result, caption=f"{rf_sel} — {len(rf_anns)} annotation(s)",
                         use_container_width=True)

        with col_rf_info:
            in_csv = rf_sel in csv_name_set
            st.markdown(f"**Image:** `{rf_sel}`")
            st.markdown(f"**In CoralNet CSV:** {'Yes' if in_csv else 'No'}")
            st.markdown(f"**Annotations:** {len(rf_anns)}")

            if rf_anns:
                # Per-class breakdown
                ann_classes = Counter()
                ann_areas = {}
                for ann in rf_anns:
                    cat_name = coco_cat_map.get(ann["category_id"], "")
                    sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cat_name) or cat_name
                    ann_classes[sc] += 1
                    ann_areas[sc] = ann_areas.get(sc, 0) + ann.get("area", 0)

                html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
                html.append("<tr><th></th><th>Code</th><th>Name</th>"
                            "<th align='right'>Polygons</th><th align='right'>Area (px)</th></tr>")
                for sc, cnt in ann_classes.most_common():
                    color = color_by_code.get(sc, "#808080")
                    meta = LABEL_META_BY_SHORT_CODE.get(sc, {})
                    name = meta.get("Name", sc)
                    area = int(ann_areas.get(sc, 0))
                    html.append(
                        f"<tr style='border-top:1px solid #eee;'>"
                        f"<td><span style='display:inline-block;width:14px;height:14px;"
                        f"background:{color};border:1px solid #555;border-radius:3px;'></span></td>"
                        f"<td>{sc}</td><td>{name}</td>"
                        f"<td align='right'>{cnt}</td>"
                        f"<td align='right'>{area:,}</td></tr>"
                    )
                html.append("</table>")
                st.markdown("".join(html), unsafe_allow_html=True)

                # Segmentation type breakdown for this image
                seg_counter = Counter()
                for ann in rf_anns:
                    s = ann.get("segmentation")
                    seg_counter["Polygon" if isinstance(s, list) else "RLE"] += 1
                st.caption(f"Seg types: {dict(seg_counter)}")

    # ---- Section 1: Class Mapping ----
    st.markdown("---")
    st.markdown("### 1. COCO Category → Short Code Mapping")
    st.markdown("Every Roboflow category must map to a CoralNet short code. "
                "Flags: **UNMAPPED** = no mapping exists, **NOT IN CSV** = "
                "short code valid but absent from CoralNet CSV (new class from biologist).")

    map_rows = []
    for cid, cname in sorted(coco_cat_map.items()):
        sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cname)
        if sc is None and cname in LABEL_META_BY_SHORT_CODE:
            sc = cname  # merged COCO: name is already the short code
        meta = LABEL_META_BY_SHORT_CODE.get(sc, {}) if sc else {}
        meta_name = meta.get("Name", "")
        in_csv = sc in csv_label_set if sc else False
        # count annotations for this category
        ann_count = sum(1 for a in coco_raw["annotations"] if a["category_id"] == cid)
        status = "OK"
        if sc is None:
            status = "UNMAPPED"
        elif not in_csv:
            status = "NOT IN CSV"
        elif (meta_name and meta_name.lower() != cname.lower()
              and cname != sc  # merged COCO uses short codes as names — no genus to compare
              and not cname.startswith("Porites")):
            status = f"NAME MISMATCH (meta: {meta_name})"
        map_rows.append({
            "COCO ID": cid,
            "COCO Name": cname,
            "Short Code": sc or "—",
            "Labelset Name": meta_name or "—",
            "In CSV": in_csv,
            "RF Annotations": ann_count,
            "Status": status,
        })

    map_df = pd.DataFrame(map_rows)
    issues = map_df[map_df["Status"] != "OK"]
    ok = map_df[map_df["Status"] == "OK"]

    if len(issues):
        st.warning(f"{len(issues)} category/ies flagged — review below.")
        # Colour-coded HTML table
        html = ["<table style='width:100%;font-size:0.85rem;border-collapse:collapse;'>"]
        html.append("<tr style='border-bottom:2px solid #999;'>"
                    "<th>ID</th><th>COCO Name</th><th>Short Code</th>"
                    "<th>Labelset Name</th><th>In CSV</th><th>RF Anns</th><th>Status</th></tr>")
        for _, r_ in map_df.iterrows():
            bg = ""
            if r_["Status"] == "UNMAPPED":
                bg = "background:#ffe0e0;"
            elif r_["Status"] == "NOT IN CSV":
                bg = "background:#fff3cd;"
            elif "MISMATCH" in r_["Status"]:
                bg = "background:#ffe0cc;"
            html.append(
                f"<tr style='border-bottom:1px solid #eee;{bg}'>"
                f"<td>{r_['COCO ID']}</td><td>{r_['COCO Name']}</td>"
                f"<td><b>{r_['Short Code']}</b></td><td>{r_['Labelset Name']}</td>"
                f"<td>{r_['In CSV']}</td><td align='right'>{r_['RF Annotations']}</td>"
                f"<td><b>{r_['Status']}</b></td></tr>"
            )
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)

        st.markdown("**Legend:** "
                    "<span style='background:#ffe0e0;padding:2px 6px;'>UNMAPPED</span> "
                    "<span style='background:#fff3cd;padding:2px 6px;'>NOT IN CSV (new class)</span> "
                    "<span style='background:#ffe0cc;padding:2px 6px;'>NAME MISMATCH</span>",
                    unsafe_allow_html=True)
    else:
        st.success("All COCO categories map correctly to CoralNet short codes.")

    with st.expander(f"All {len(ok)} OK mappings"):
        st.dataframe(ok, width="stretch", hide_index=True)

    # ---- Section 2: Image Name Matching ----
    st.markdown("---")
    st.markdown("### 2. Image Name Matching (Roboflow → CoralNet)")
    st.markdown("Roboflow mangles filenames. We de-mangle via `extra.name` then normalize. "
                "**Unmatched** = image in Roboflow but NOT in the CoralNet CSV.")

    matched_imgs_list = []
    unmatched_imgs_list = []
    for iid, info in coco_img_meta.items():
        entry = {
            "COCO ID": iid,
            "Roboflow file_name": info["file_name"],
            "extra.name": info["extra_name"],
            "De-mangled": info["demangled"],
            "Normalized": info["norm"],
            "W": info["width"],
            "H": info["height"],
            "Tags": ", ".join(info["tags"]) if info["tags"] else "",
        }
        if info["norm"] in csv_name_set:
            matched_imgs_list.append(entry)
        else:
            unmatched_imgs_list.append(entry)

    c1, c2, c3 = st.columns(3)
    c1.metric("Matched", f"{len(matched_imgs_list):,}")
    c2.metric("Unmatched", f"{len(unmatched_imgs_list):,}")
    c3.metric("Match rate", f"{100 * len(matched_imgs_list) / max(len(coco_img_meta), 1):.1f}%")

    if unmatched_imgs_list:
        st.warning(f"{len(unmatched_imgs_list)} Roboflow images have NO match in the CoralNet CSV. "
                   "These are likely new images annotated by the biologist.")
        unmatched_df = pd.DataFrame(unmatched_imgs_list)

        # Show prefix distribution
        prefix_counts = Counter(r["Normalized"].split("_")[0] if "_" in r["Normalized"]
                                else r["Normalized"][:6] for r in unmatched_imgs_list)
        st.markdown("**Filename prefix distribution (unmatched):**")
        prefix_str = " · ".join(f"`{k}` ({v})" for k, v in prefix_counts.most_common(15))
        st.markdown(prefix_str)

        # Count annotations lost
        unmatched_ids = {r["COCO ID"] for r in unmatched_imgs_list}
        lost_anns = sum(1 for a in coco_raw["annotations"] if a["image_id"] in unmatched_ids)
        st.info(f"**{lost_anns:,}** polygon annotations sit on unmatched images "
                f"({100 * lost_anns / max(len(coco_raw['annotations']), 1):.1f}% of total RF annotations).")

        with st.expander(f"Full list of {len(unmatched_imgs_list)} unmatched images"):
            st.dataframe(unmatched_df, width="stretch", hide_index=True, height=400)
    else:
        st.success("All Roboflow images match a CoralNet CSV image.")

    with st.expander(f"Sample of {min(20, len(matched_imgs_list))} matched images (spot-check)"):
        st.dataframe(pd.DataFrame(matched_imgs_list[:20]), width="stretch", hide_index=True)

    # ---- Section 3: De-mangling audit ----
    st.markdown("---")
    st.markdown("### 3. Filename De-mangling Audit")
    st.markdown("Show every image where de-mangling **changed** the name, so you can verify "
                "the regex is correct.")

    changed = [info for info in coco_img_meta.values()
               if info["extra_name"] != info["demangled"]]
    unchanged = [info for info in coco_img_meta.values()
                 if info["extra_name"] == info["demangled"]]

    c1, c2 = st.columns(2)
    c1.metric("De-mangled (changed)", len(changed))
    c2.metric("Unchanged", len(unchanged))

    if changed:
        demangle_df = pd.DataFrame([{
            "extra.name": i["extra_name"],
            "De-mangled": i["demangled"],
            "Normalized": i["norm"],
            "Matched CSV": i["norm"] in csv_name_set,
        } for i in changed[:100]])
        st.dataframe(demangle_df, width="stretch", hide_index=True, height=350)
        if len(changed) > 100:
            st.caption(f"Showing first 100 of {len(changed)} changed names.")

    # ---- Section 4: Annotation integrity ----
    st.markdown("---")
    st.markdown("### 4. Annotation Integrity")

    seg_types = Counter()
    zero_area = 0
    for a in coco_raw["annotations"]:
        s = a.get("segmentation")
        if isinstance(s, dict):
            seg_types["RLE"] += 1
        elif isinstance(s, list):
            seg_types["Polygon"] += 1
        else:
            seg_types["Other"] += 1
        if a.get("area", 0) == 0:
            zero_area += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total annotations", f"{len(coco_raw['annotations']):,}")
    c2.metric("Polygon segmentations", seg_types.get("Polygon", 0))
    c3.metric("RLE segmentations", seg_types.get("RLE", 0))
    c4.metric("Area = 0", zero_area)

    if zero_area:
        st.warning(f"{zero_area} annotation(s) have area=0 — these are degenerate polygons.")
    if seg_types.get("Other", 0):
        st.error(f"{seg_types['Other']} annotation(s) have unknown segmentation format.")

    # Images with no annotations
    annotated_ids = set(a["image_id"] for a in coco_raw["annotations"])
    no_ann = [coco_img_meta[iid] for iid in coco_img_meta if iid not in annotated_ids]
    if no_ann:
        st.warning(f"{len(no_ann)} COCO image(s) have **zero annotations**.")
        st.dataframe(pd.DataFrame([{
            "Normalized": i["norm"],
            "extra.name": i["extra_name"],
            "Tags": ", ".join(i["tags"]) if i["tags"] else "",
        } for i in no_ann]), width="stretch", hide_index=True)

    # Area distribution
    areas = [a["area"] for a in coco_raw["annotations"] if a["area"] > 0]
    if areas:
        st.markdown("**Polygon area distribution (px²):**")
        area_df = pd.DataFrame({"area": areas})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{min(areas):,.0f}")
        c2.metric("Median", f"{np.median(areas):,.0f}")
        c3.metric("Mean", f"{np.mean(areas):,.0f}")
        c4.metric("Max", f"{max(areas):,.0f}")

    # ---- Section 5: Roboflow-only vs CoralNet-only classes ----
    st.markdown("---")
    st.markdown("### 5. Class Coverage Comparison")

    rf_codes_used = set()
    for a in coco_raw["annotations"]:
        cname = coco_cat_map.get(a["category_id"], "")
        sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cname)
        if sc is None and cname in LABEL_META_BY_SHORT_CODE:
            sc = cname
        if sc:
            rf_codes_used.add(sc)

    both = csv_label_set & rf_codes_used
    cn_only = csv_label_set - rf_codes_used
    rf_only = rf_codes_used - csv_label_set

    c1, c2, c3 = st.columns(3)
    c1.metric("In both sources", len(both))
    c2.metric("CoralNet only", len(cn_only))
    c3.metric("Roboflow only", len(rf_only))

    if rf_only:
        st.info("**Roboflow-only classes** (annotated by biologist, absent from CoralNet CSV — "
                "these are new training classes):")
        rf_only_rows = []
        for sc in sorted(rf_only):
            meta = LABEL_META_BY_SHORT_CODE.get(sc, {})
            count = sum(1 for a in coco_raw["annotations"]
                        if (_ROBOFLOW_NAME_TO_SHORT_CODE.get(coco_cat_map.get(a["category_id"], ""))
                    or coco_cat_map.get(a["category_id"], "")) == sc)
            rf_only_rows.append({
                "Short Code": sc,
                "Name": meta.get("Name", sc),
                "Functional Group": meta.get("Functional Group", "Unknown"),
                "RF Annotations": count,
            })
        st.dataframe(pd.DataFrame(rf_only_rows), width="stretch", hide_index=True)

    if cn_only:
        with st.expander(f"CoralNet-only classes ({len(cn_only)} — no Roboflow polygons)"):
            cn_only_rows = []
            for sc in sorted(cn_only):
                meta = LABEL_META_BY_SHORT_CODE.get(sc, {})
                count = int(df[df["Label"] == sc].shape[0])
                cn_only_rows.append({
                    "Short Code": sc,
                    "Name": meta.get("Name", sc),
                    "Functional Group": meta.get("Functional Group", "Unknown"),
                    "CN Points": count,
                })
            st.dataframe(pd.DataFrame(cn_only_rows), width="stretch", hide_index=True)

    # ---- Section 6: Per-category annotation count comparison ----
    st.markdown("---")
    st.markdown("### 6. Per-Category Annotation Counts (raw COCO)")
    st.markdown("Direct count from the COCO JSON — verify nothing was lost during loading.")

    raw_cat_counts = Counter()
    for a in coco_raw["annotations"]:
        raw_cat_counts[a["category_id"]] += 1

    raw_rows = []
    for cid in sorted(raw_cat_counts.keys()):
        cname = coco_cat_map.get(cid, "???")
        sc = _ROBOFLOW_NAME_TO_SHORT_CODE.get(cname)
        if sc is None:
            sc = cname if cname in LABEL_META_BY_SHORT_CODE else "—"
        # Compare with what load_coco_annotations produced
        loaded_count = len(df_rf[df_rf["Label"] == sc]) if sc != "—" and len(df_rf) else 0
        raw_count = raw_cat_counts[cid]
        match = raw_count == loaded_count
        raw_rows.append({
            "COCO ID": cid,
            "Category": cname,
            "Short Code": sc,
            "Raw COCO count": raw_count,
            "Loaded count": loaded_count,
            "Match": "yes" if match else f"MISMATCH (diff={raw_count - loaded_count})",
        })

    raw_df = pd.DataFrame(raw_rows)
    mismatches = raw_df[~raw_df["Match"].str.startswith("yes")]
    if len(mismatches):
        st.warning(f"{len(mismatches)} category/ies have count mismatches between raw COCO and loaded data. "
                   "This is expected when unmatched images are excluded or categories are unmapped.")
    st.dataframe(raw_df, width="stretch", hide_index=True)

    # ---- Section 7: Duplicate normalized names ----
    st.markdown("---")
    st.markdown("### 7. Duplicate Normalized Names")
    norm_counts = Counter(info["norm"] for info in coco_img_meta.values())
    dupes = {k: v for k, v in norm_counts.items() if v > 1}
    if dupes:
        st.error(f"{len(dupes)} normalized image name(s) appear more than once — "
                 "this would cause data collision!")
        dupe_rows = []
        for name, cnt in sorted(dupes.items(), key=lambda x: -x[1]):
            variants = [info for info in coco_img_meta.values() if info["norm"] == name]
            for v in variants:
                dupe_rows.append({
                    "Normalized": name,
                    "file_name": v["file_name"],
                    "extra.name": v["extra_name"],
                    "Count": cnt,
                })
        st.dataframe(pd.DataFrame(dupe_rows), width="stretch", hide_index=True)
    else:
        st.success("No duplicate normalized names — each COCO image maps to a unique identifier.")
