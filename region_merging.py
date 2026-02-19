"""
Region merging postprocessor for graph-based segmentation.

Merges fragmented per-class regions into coherent object-level masks suitable
for COCO instance segmentation export.  Designed to run *after* label
assignment (the labeled_mask already has class IDs) and *before* COCO export.

Pipeline:
    1. Remove speckle regions below min_area.
    2. Merge small same-class regions into their nearest same-class neighbor.
    3. Optionally merge adjacent same-class regions whose mean colors are
       similar (color_threshold).
    4. Fill remaining small gaps via morphological closing.

All thresholds are configurable.  The module never touches the upstream
Felzenszwalb implementation.
"""

import numpy as np
import cv2
from scipy import ndimage


def merge_regions(labeled_mask, image=None,
                  min_area=100,
                  small_region_merge=500,
                  color_threshold=30.0,
                  morph_close_ksize=5):
    """
    Merge fragmented regions in a labeled class mask into coherent objects.

    Args:
        labeled_mask : 2-D uint8 array with class IDs (0 = background).
        image        : (optional) BGR image at the same resolution as the mask.
                       Required only when color_threshold > 0.
        min_area     : Remove connected components smaller than this (pixels).
        small_region_merge : Merge same-class components smaller than this into
                       their nearest same-class neighbor.
        color_threshold : Max Euclidean RGB distance for merging adjacent
                       same-class regions.  0 = skip color-based merging.
        morph_close_ksize : Kernel size for morphological closing to fill tiny
                       gaps inside objects.  0 = skip.

    Returns:
        merged_mask : 2-D uint8 array with the same class IDs, fewer regions.
    """
    merged = labeled_mask.copy()

    # --- Step 1: morphological closing per class to fill small holes ----------
    if morph_close_ksize > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_close_ksize, morph_close_ksize))
        for cid in np.unique(merged):
            if cid == 0:
                continue
            binary = (merged == cid).astype(np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            # Only fill new pixels that are currently background
            fill = (closed > 0) & (merged == 0)
            merged[fill] = cid

    # --- Step 2: remove speckle (tiny) regions --------------------------------
    merged = _remove_small_components(merged, min_area)

    # --- Step 3: merge small same-class regions into nearest neighbor ----------
    if small_region_merge > min_area:
        merged = _merge_small_into_neighbors(merged, small_region_merge)

    # --- Step 4: color-similarity merge of adjacent same-class regions --------
    if color_threshold > 0 and image is not None:
        merged = _merge_by_color(merged, image, color_threshold)

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_small_components(mask, min_area):
    """Set connected components smaller than min_area to 0 (background)."""
    out = mask.copy()
    for cid in np.unique(out):
        if cid == 0:
            continue
        binary = (out == cid).astype(np.uint8)
        n_comp, labels = cv2.connectedComponents(binary)
        for comp in range(1, n_comp):
            comp_mask = labels == comp
            if comp_mask.sum() < min_area:
                out[comp_mask] = 0
    return out


def _merge_small_into_neighbors(mask, threshold):
    """Merge small same-class components into their nearest same-class neighbor."""
    out = mask.copy()

    for cid in np.unique(out):
        if cid == 0:
            continue

        binary = (out == cid).astype(np.uint8)
        n_comp, labels = cv2.connectedComponents(binary)
        if n_comp <= 2:
            # 0 or 1 real component -> nothing to merge
            continue

        # Compute area for each component
        areas = ndimage.sum(binary, labels, range(1, n_comp))

        # Identify large components (merge targets)
        large_ids = [i + 1 for i, a in enumerate(areas) if a >= threshold]
        small_ids = [i + 1 for i, a in enumerate(areas) if a < threshold]

        if not large_ids or not small_ids:
            continue

        # Build a distance map from all large-component pixels
        large_mask = np.isin(labels, large_ids).astype(np.uint8)
        dist, nearest_idx = cv2.distanceTransformWithLabels(
            1 - large_mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

        # nearest_idx gives a pixel-level label; map it back to component id
        # Build lookup: pixel label -> component id
        # nearest_idx labels are 1-indexed pixel IDs in the *large_mask* region
        large_ys, large_xs = np.where(large_mask > 0)
        if len(large_ys) == 0:
            continue

        # For each small component, find the component ID of the nearest large one
        for sid in small_ids:
            small_pixels = labels == sid
            # Get the nearest-large-pixel indices for all small pixels
            sy, sx = np.where(small_pixels)
            if len(sy) == 0:
                continue
            # Use the pixel nearest to the centroid as representative
            cy, cx = int(sy.mean()), int(sx.mean())
            cy = np.clip(cy, 0, mask.shape[0] - 1)
            cx = np.clip(cx, 0, mask.shape[1] - 1)

            # Search expanding rings from centroid for a large-component pixel
            search_radius = int(dist[cy, cx]) + 5
            y_lo = max(0, cy - search_radius)
            y_hi = min(mask.shape[0], cy + search_radius + 1)
            x_lo = max(0, cx - search_radius)
            x_hi = min(mask.shape[1], cx + search_radius + 1)

            crop_large = large_mask[y_lo:y_hi, x_lo:x_hi]
            if crop_large.sum() == 0:
                continue  # no large neighbor close enough

            # This small component stays the same class (already cid) so the
            # COCO exporter will naturally merge them if they touch after closing.
            # But we do an explicit spatial merge: relabel small pixels into the
            # large component's label space by just keeping them as cid (they
            # already are).  The key is: they're already the same class.  The
            # problem is they're *disconnected*.  So we draw a thin bridge.
            crop_labels = labels[y_lo:y_hi, x_lo:x_hi]
            crop_large_pixels = np.argwhere(np.isin(crop_labels, large_ids))
            if len(crop_large_pixels) == 0:
                continue
            # Find closest large pixel to centroid
            dists = np.abs(crop_large_pixels[:, 0] - (cy - y_lo)) + \
                    np.abs(crop_large_pixels[:, 1] - (cx - x_lo))
            nearest = crop_large_pixels[dists.argmin()]
            target_y = nearest[0] + y_lo
            target_x = nearest[1] + x_lo

            # Draw a thin bridge (2px) between centroid and nearest large pixel
            bridge = np.zeros(mask.shape[:2], dtype=np.uint8)
            cv2.line(bridge, (cx, cy), (target_x, target_y), 1, thickness=2)
            # Only fill bridge where background
            bridge_fill = (bridge > 0) & (out == 0)
            out[bridge_fill] = cid

    return out


def _merge_by_color(mask, image, color_threshold):
    """Merge adjacent same-class components with similar mean color."""
    out = mask.copy()
    # Convert image to float for distance computation
    img_f = image.astype(np.float32)

    for cid in np.unique(out):
        if cid == 0:
            continue

        binary = (out == cid).astype(np.uint8)
        n_comp, labels = cv2.connectedComponents(binary)
        if n_comp <= 2:
            continue

        # Compute mean color per component
        mean_colors = {}
        for comp in range(1, n_comp):
            pixels = img_f[labels == comp]
            if len(pixels) == 0:
                continue
            mean_colors[comp] = pixels.mean(axis=0)

        # Build adjacency via dilation
        adjacency = _build_adjacency(labels, n_comp)

        # Greedy merge: union-find
        parent = {i: i for i in range(1, n_comp)}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for (a, b) in adjacency:
            if a not in mean_colors or b not in mean_colors:
                continue
            dist = np.linalg.norm(mean_colors[a] - mean_colors[b])
            if dist < color_threshold:
                union(a, b)

        # Relabel: components that share a root get merged spatially
        # (they are already the same class, we just need them connected
        #  for the COCO exporter to treat them as one instance)
        groups = {}
        for comp in range(1, n_comp):
            root = find(comp)
            groups.setdefault(root, []).append(comp)

        # For each merged group, fill gaps between components with closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        for root, members in groups.items():
            if len(members) <= 1:
                continue
            group_mask = np.isin(labels, members).astype(np.uint8)
            closed = cv2.morphologyEx(group_mask, cv2.MORPH_CLOSE, kernel)
            fill = (closed > 0) & (out == 0)
            out[fill] = cid

    return out


def _build_adjacency(labels, n_comp):
    """Find pairs of components that are adjacent (share a border). Vectorized."""
    adj = set()
    # Horizontal neighbors
    left = labels[:, :-1].ravel()
    right = labels[:, 1:].ravel()
    diff_h = left != right
    a_h, b_h = left[diff_h], right[diff_h]
    valid = (a_h > 0) & (b_h > 0)
    a_h, b_h = a_h[valid], b_h[valid]
    # Canonical order
    lo = np.minimum(a_h, b_h)
    hi = np.maximum(a_h, b_h)
    pairs_h = np.unique(np.column_stack([lo, hi]), axis=0)
    for row in pairs_h:
        adj.add((int(row[0]), int(row[1])))

    # Vertical neighbors
    top = labels[:-1, :].ravel()
    bot = labels[1:, :].ravel()
    diff_v = top != bot
    a_v, b_v = top[diff_v], bot[diff_v]
    valid = (a_v > 0) & (b_v > 0)
    a_v, b_v = a_v[valid], b_v[valid]
    lo = np.minimum(a_v, b_v)
    hi = np.maximum(a_v, b_v)
    pairs_v = np.unique(np.column_stack([lo, hi]), axis=0)
    for row in pairs_v:
        adj.add((int(row[0]), int(row[1])))

    return adj
