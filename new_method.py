"""Depth-bracketed lens-blur preprocessor.

Given an input image, this script:
  1. Generates a depth map using OpenCV's DNN module (MiDaS small ONNX).
  2. Brackets the depth map into DEPTH_COUNT discrete luminance bins.
  3. For every depth slice i in [0, DEPTH_COUNT) it writes:
        {i}_alpha.png   - white where the slice is, black elsewhere
        {i}_masked.png  - the source image kept only inside the slice
        {i}_blurred.png - lens-blurred {i}_masked (premultiplied colour, before
                          dividing by blurred alpha / unmult)
        {i}_unmult.png  - lens-blurred {i}_masked / lens-blurred {i}_alpha
                          (0/0 := 0), which "unpremultiplies" the slice so all
                          non-zero values are restored to full brightness.

All outputs land in ./temp-images.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np


# ----- Configuration ---------------------------------------------------------

INPUT_IMAGE = "images/lego_flowers.jpeg"
TEMP_DIR = "./temp-images"
BLUR_STRENGTH = 0.4
DEPTH_COUNT = 8        # number of depth slices to bracket the depth map into

# Per-slice lens blur radius is interpolated between these two values based on
# the slice's depth. MiDaS uses brighter == closer, so the closest slice gets
# MIN_BLUR_RADIUS and the farthest slice gets MAX_BLUR_RADIUS.
MIN_BLUR_RADIUS = 0    # camera lens (disk) blur radius for the nearest slice
MAX_BLUR_RADIUS = 60   # camera lens (disk) blur radius for the farthest slice

# Number of top (= closest) slices that are treated as "in focus" -- their
# blur radius is forced to 0, the highlight boost is bypassed, and they are
# composited straight from the source image through their hard alpha. The
# remaining DEPTH_COUNT - IN_FOCUS_LAYERS slices use the bokeh pipeline.
IN_FOCUS_LAYERS = 2

# Anything smaller than this (after blur) is considered "zero" for the
# unpremultiply step. blurred_alpha is in [0, 1]; blurred_masked is compared
# in the same normalized space (i.e. /255) so a single epsilon covers both.
UNMULT_EPSILON = 1e-3

# MiDaS ONNX model. v2.1 is the only version Intel ships as ONNX, so we can
# load it through cv2.dnn with no extra dependencies. Choose:
#   "small" -> 65 MB, 256x256 input, very fast, lower detail.
#   "large" -> 400 MB, 384x384 input, slower, ~10% more accurate and much
#              sharper edges. Best quality reachable without pulling in torch.
# (MiDaS v3 / v3.1 are .pt only -- using them would require torch + timm +
# the upstream MiDaS source tree.)
MIDAS_VARIANT = "large"

_MIDAS_VARIANTS = {
    "small": {
        "path": "models/midas_v21_small.onnx",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "input_size": 256,
    },
    "large": {
        "path": "models/midas_v21_large.onnx",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.onnx",
        "input_size": 384,
    },
}
MIDAS_MODEL_PATH = _MIDAS_VARIANTS[MIDAS_VARIANT]["path"]
MIDAS_MODEL_URL = _MIDAS_VARIANTS[MIDAS_VARIANT]["url"]
MIDAS_INPUT_SIZE = _MIDAS_VARIANTS[MIDAS_VARIANT]["input_size"]


# ----- Depth estimation ------------------------------------------------------

def _ensure_midas_model() -> None:
    """Download the MiDaS small ONNX model the first time it is needed."""
    os.makedirs(os.path.dirname(MIDAS_MODEL_PATH), exist_ok=True)
    if os.path.exists(MIDAS_MODEL_PATH):
        return
    print(f"Downloading MiDaS model -> {MIDAS_MODEL_PATH}")
    urllib.request.urlretrieve(MIDAS_MODEL_URL, MIDAS_MODEL_PATH)
    print("MiDaS model downloaded.")


def generate_depth_map(image_bgr: np.ndarray) -> np.ndarray:
    """Run MiDaS small via cv2.dnn and return an 8-bit depth map.

    Brighter pixels = closer to the camera (standard MiDaS convention).
    """
    _ensure_midas_model()
    net = cv2.dnn.readNet(MIDAS_MODEL_PATH)

    h, w = image_bgr.shape[:2]

    # MiDaS expects a square RGB input (256 for small, 384 for large),
    # normalized with ImageNet mean/std.
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0 / 255.0,
        size=(MIDAS_INPUT_SIZE, MIDAS_INPUT_SIZE),
        mean=(0.485, 0.456, 0.406),
        swapRB=True,
        crop=False,
    )
    # blobFromImage doesn't support per-channel std, so apply it ourselves.
    blob[:, 0, :, :] /= 0.229
    blob[:, 1, :, :] /= 0.224
    blob[:, 2, :, :] /= 0.225

    net.setInput(blob)
    depth = net.forward().squeeze()
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)
    return (depth * 255.0).astype(np.uint8)


# ----- Highlight expansion ---------------------------------------------------

# Bokeh highlight boost. f(x) = -700/(x - 260) + x is roughly identity for low
# values and ramps sharply upward near 255 (e.g. f(200) ~= 212, f(250) = 320,
# f(255) = 395). Lifting bright pixels into HDR space before the lens blur is
# what gives blurred highlights their characteristic punchy bokeh discs --
# without it, a 255 pixel just smears out to a dim grey blob.
HIGHLIGHT_NUMERATOR = -700.0
HIGHLIGHT_OFFSET = 260.0

# Swirly-bokeh radial pre-warp. SWIRL_AMOUNT == 0 -> identity (both curves
# collapse to f(x) = g(x) = x). Larger values produce more aggressive radial
# stretching during the blur and therefore more pronounced swirl in the
# resulting bokeh discs. Try 0.1 - 0.6.
SWIRL_AMOUNT = 0.4

# Extra safety margin on top of the analytic diverge-pad ratio. 1.0 is exactly
# tight (the converge sampler lands right on the padded corner); a touch over
# 1 keeps the bilinear sampler from grazing the very edge.
SWIRL_PAD_SAFETY = 1.02


# Per-slice highlight bleed ramps linearly from MIN_HIGHLIGHT_STRENGTH at the
# closest slice to MAX_HIGHLIGHT_STRENGTH at the farthest slice. Setting both
# to 1.0 reproduces the old "every blurred layer gets the full curve, every
# in-focus layer gets none" behaviour as a hard step. Nonzero MIN means even
# the in-focus layer picks up a touch of bloom, so a streetlight in focus
# still glows a bit instead of looking surgically clipped.
MIN_HIGHLIGHT_STRENGTH = 0.25
MAX_HIGHLIGHT_STRENGTH = 1.0


def expand_highlights(image_f: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply the bokeh highlight curve, scaled by `strength` in [0, 1].

    strength == 0 -> identity (no boost), strength == 1 -> full curve.
    Input/output are float, unbounded.
    """
    if strength <= 0.0:
        return image_f
    return strength * (HIGHLIGHT_NUMERATOR / (image_f - HIGHLIGHT_OFFSET)) + image_f


# ----- Radial swirl warp (divergence / convergence) --------------------------

# These two curves are exact inverses of each other (f(g(x)) = g(f(x)) = x):
#
#       f(x) = ((x + a/(1 - x)) - a) / (a + 1)         {a > 0, x < 1}
#       g(x) = 1/2 * ( -sqrt(b**2 + 2b((x(b+1)) + 1)
#                            + ((x(b+1)) - 1)**2)
#                      + b + (x(b+1)) + 1 )            {b == a}
#
# f spreads radii outward (divergence); g pulls them back (convergence). We
# apply f to the source before blurring and g to the final composite, so the
# blur happens in a stretched coordinate frame -- which is what produces the
# off-axis "swirly bokeh" you get from lenses with strong field curvature
# (Helios 44, Petzval, etc.).

def _f_diverge(r: np.ndarray, a: float) -> np.ndarray:
    return ((r + a / (1.0 - r)) - a) / (a + 1.0)


def _g_converge(r: np.ndarray, a: float) -> np.ndarray:
    b = a
    t = r * (b + 1.0)
    inside = b * b + 2.0 * b * (t + 1.0) + (t - 1.0) ** 2
    return 0.5 * (-np.sqrt(inside) + b + t + 1.0)


def _build_radial_remap(shape, transform):
    """Build cv2.remap arrays for a radial warp.

    `transform(r_out)` returns the source radius (both normalized so that the
    image's half-diagonal == 1). Pixel (x, y) in the output samples from the
    point on the same ray from the centre at the transformed radius.
    """
    h, w = shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    half_diag = float(np.hypot(cx, cy))

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    r_out = np.sqrt(dx * dx + dy * dy) / half_diag

    # Clamp away from 1 so f(x) = a/(1-x) never explodes for the corners.
    r_out_clamped = np.clip(r_out, 0.0, 0.999)
    r_in = transform(r_out_clamped)

    # scale = r_in / r_out gives the per-pixel radial sampling offset.
    # At r_out == 0 the sample point is the centre regardless of scale, so
    # fall back to 1.0 there to avoid 0/0.
    scale = np.where(r_out > 1e-8, r_in / np.maximum(r_out, 1e-8), 1.0)

    map_x = (cx + dx * scale).astype(np.float32)
    map_y = (cy + dy * scale).astype(np.float32)
    return map_x, map_y


def diverge(image: np.ndarray, a: float) -> np.ndarray:
    """Push pixels outward from the centre (pre-blur swirl)."""
    if a <= 0.0:
        return image.copy()
    map_x, map_y = _build_radial_remap(image.shape, lambda r: _g_converge(r, a))
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def converge(image: np.ndarray, a: float) -> np.ndarray:
    """Pull pixels back toward the centre (post-blur swirl undo)."""
    if a <= 0.0:
        return image.copy()
    map_x, map_y = _build_radial_remap(image.shape, lambda r: _f_diverge(r, a))
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ----- Lens blur -------------------------------------------------------------

# Onion-ring bokeh. Real aspheric lens elements show concentric brightness
# rings inside the bokeh disc, plus a hot rim at the edge (the "cat's eye"
# soap-bubble look). We model this as a radial weight on top of the flat disk:
#
#     w(r) = 1 + EDGE_BOOST * (r / R)^EDGE_FALLOFF
#                + RING_AMPLITUDE * cos(2*pi * RING_COUNT * r / R)^2
#
# where r is the distance from the kernel centre and R is the kernel radius.
# Set ONION_RING_AMOUNT = 0 to recover the original perfect disk.
ONION_RING_AMOUNT = 1.0     # global scale on the non-flat component (0 disables)
EDGE_BOOST = 0.6            # extra weight at the very rim (0 = flat to the edge)
EDGE_FALLOFF = 4.0          # how concentrated the rim brightening is (higher = thinner ring)
RING_COUNT = 2.5            # number of concentric rings inside the disc
RING_AMPLITUDE = 0.15       # depth of the inner ring modulation


def _disk_kernel(radius: int) -> np.ndarray:
    """Disk kernel with an optional onion-ring radial weighting.

    The base shape is still a hard-edged disk of the given pixel radius (so
    the bokeh-disc footprint matches the perfect-disk version exactly); the
    onion-ring effect comes from a smooth radial weight that brightens the
    rim and adds a couple of concentric bands inside.
    """
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    rr = np.sqrt(x * x + y * y).astype(np.float32)
    mask = rr <= float(radius)

    if ONION_RING_AMOUNT <= 0.0 or radius == 0:
        kernel = mask.astype(np.float32)
    else:
        r_norm = np.clip(rr / float(radius), 0.0, 1.0)
        rim = EDGE_BOOST * np.power(r_norm, EDGE_FALLOFF)
        rings = RING_AMPLITUDE * np.cos(np.pi * RING_COUNT * r_norm) ** 2
        weight = 1.0 + ONION_RING_AMOUNT * (rim + rings)
        kernel = (mask.astype(np.float32)) * weight.astype(np.float32)

    kernel /= kernel.sum()
    return kernel


def lens_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Convolve `image` with a disk kernel (camera lens / bokeh-style blur)."""
    if radius <= 0:
        return image.copy()
    kernel = _disk_kernel(radius)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


# ----- Main pipeline ---------------------------------------------------------

def main() -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)

    image_bgr = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {INPUT_IMAGE}")

    print("Generating depth map ...")
    depth = generate_depth_map(image_bgr)
    cv2.imwrite(os.path.join(TEMP_DIR, "depth.png"), depth)

    # Pre-blur "divergence" warp. We diverge BOTH the source image and the
    # depth map (so every layer's mask still aligns with the warped pixels),
    # run the entire bokeh pipeline in this stretched coordinate frame, and
    # finally apply the inverse "convergence" warp to the composited canvas.
    # The blur happens in diverged space, which is what gives off-axis bokeh
    # discs their swirly elongation.
    #
    # We pad the canvas before diverging and crop after converging so that the
    # converge step only ever samples real (post-diverge) pixels -- otherwise
    # f explodes near the image edge and the corners get filled by whatever
    # cv2.remap's borderMode chooses, which shows up as tiled / mirrored junk.
    # The padding factor is set so that, in the padded frame, the original
    # image corner sits at normalized radius r* with f(r*, a) == 1; that is
    # exactly r* = g(1, a). So D'/D = 1 / g(1, a), plus a small safety margin.
    orig_h, orig_w = image_bgr.shape[:2]
    pad_x = pad_y = 0
    if SWIRL_AMOUNT > 0.0:
        pad_ratio = SWIRL_PAD_SAFETY / float(_g_converge(np.array([1.0]), SWIRL_AMOUNT)[0])
        pad_x = int(np.ceil(orig_w * (pad_ratio - 1.0) * 0.5))
        pad_y = int(np.ceil(orig_h * (pad_ratio - 1.0) * 0.5))
        image_bgr = cv2.copyMakeBorder(
            image_bgr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT,
        )
        depth = cv2.copyMakeBorder(
            depth, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT,
        )
        print(
            f"Diverging source + depth (a = {SWIRL_AMOUNT}, "
            f"padded {orig_w}x{orig_h} -> {image_bgr.shape[1]}x{image_bgr.shape[0]}) ..."
        )
        image_bgr = diverge(image_bgr, SWIRL_AMOUNT)
        depth = diverge(depth, SWIRL_AMOUNT)
        cv2.imwrite(os.path.join(TEMP_DIR, "depth_diverged.png"), depth)
        cv2.imwrite(os.path.join(TEMP_DIR, "image_diverged.png"), image_bgr)

    # Single float copy of the source. The highlight boost is now applied
    # per-slice with a smoothly-ramped strength (see expand_highlights and
    # MIN/MAX_HIGHLIGHT_STRENGTH) instead of as a binary "boosted vs not"
    # split. Everything downstream stays in float32 with no clamping; we
    # only clip back to [0, 255] when writing the final 8-bit PNG.
    image_orig_f = image_bgr.astype(np.float32)
    bin_edges = np.linspace(0.0, 256.0, DEPTH_COUNT + 1)

    # Slices with index >= focus_threshold are in focus (radius 0, no boost).
    focus_threshold = DEPTH_COUNT - IN_FOCUS_LAYERS

    # Final composite canvas. We composite back-to-front: in our depth
    # convention (MiDaS: brighter == closer), slice 0 is the farthest layer
    # and slice DEPTH_COUNT - 1 is the closest, so a simple ascending loop
    # paints farther layers first and lets nearer layers cover them.
    canvas = np.zeros_like(image_orig_f)

    for i in range(DEPTH_COUNT):
        lo, hi = bin_edges[i], bin_edges[i + 1]

        # In-focus slices: force radius to 0 (the highlight boost is still
        # applied at MIN_HIGHLIGHT_STRENGTH so there's no hard discontinuity).
        # Bokeh slices: interpolate radius based on mid-depth (brighter ==
        # closer in MiDaS, so closer slices get smaller radii).
        in_focus = i >= focus_threshold
        if in_focus:
            radius = 0
        else:
            # Shift the blur curve back so the in-focus group "absorbs" the
            # closest IN_FOCUS_LAYERS - 1 slots. The nearest unfocused layer
            # then receives the blur that the second-closest layer would have
            # received in the original (IN_FOCUS_LAYERS = 1) scheme, the next
            # one back gets the third-closest layer's blur, and so on.
            effective_i = min(i + (IN_FOCUS_LAYERS - 1), DEPTH_COUNT - 1)
            lo_eff, hi_eff = bin_edges[effective_i], bin_edges[effective_i + 1]
            closeness = (lo_eff + hi_eff) * 0.5 / 255.0
            radius = int(round(
                (MIN_BLUR_RADIUS
                 + (MAX_BLUR_RADIUS - MIN_BLUR_RADIUS) * (1.0 - closeness)) / (1 / BLUR_STRENGTH)
            ))
            radius = max(radius, 0)

        # Highlight bleed strength ramps linearly from MAX (farthest, i = 0)
        # to MIN (closest, i = DEPTH_COUNT - 1). All slices -- including
        # in-focus ones -- get at least MIN_HIGHLIGHT_STRENGTH of bloom, so
        # bright pixels never abruptly stop blooming as they cross focus.
        depth_t = i / max(DEPTH_COUNT - 1, 1)  # 0 at farthest, 1 at closest
        highlight_strength = (
            MAX_HIGHLIGHT_STRENGTH
            + (MIN_HIGHLIGHT_STRENGTH - MAX_HIGHLIGHT_STRENGTH) * depth_t
        )
        slice_source_f = expand_highlights(image_orig_f, highlight_strength)

        # Build the alpha matte for this slice. Make the last bin inclusive on
        # the high end so depth==255 isn't dropped.
        if i == DEPTH_COUNT - 1:
            mask = (depth >= lo) & (depth <= hi)
        else:
            mask = (depth >= lo) & (depth < hi)
        mask_f = mask.astype(np.float32)

        alpha_uint8 = (mask_f * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(TEMP_DIR, f"{i}_alpha.png"), alpha_uint8)

        # Premultiplied (masked) image: source kept only inside the slice.
        masked_f = slice_source_f * mask_f[..., None]
        cv2.imwrite(
            os.path.join(TEMP_DIR, f"{i}_masked.png"),
            np.clip(masked_f, 0.0, 255.0).astype(np.uint8),
        )

        # Lens-blur both, then unpremultiply: blurred_color / blurred_alpha.
        blurred_masked = lens_blur(masked_f, radius)
        blurred_alpha = lens_blur(mask_f, radius)
        cv2.imwrite(
            os.path.join(TEMP_DIR, f"{i}_blurred.png"),
            np.clip(blurred_masked, 0.0, 255.0).astype(np.uint8),
        )

        # If both numerator and denominator are essentially zero (i.e. the
        # only "value" present is float / convolution residue), the divide is
        # an indeterminate 0/0 -- return 0. Anywhere either side carries real
        # signal, do the straight unpremultiply.
        alpha_is_zero = blurred_alpha < UNMULT_EPSILON
        masked_is_zero = blurred_masked.max(axis=-1) / 255.0 < UNMULT_EPSILON
        both_zero = alpha_is_zero & masked_is_zero

        with np.errstate(divide="ignore", invalid="ignore"):
            unmult = blurred_masked / blurred_alpha[..., None]
        unmult = np.where(both_zero[..., None], 0.0, unmult)

        cv2.imwrite(
            os.path.join(TEMP_DIR, f"{i}_unmult.png"),
            np.clip(unmult, 0.0, 255.0).astype(np.uint8),
        )

        # ---- Adjusted alpha matte ------------------------------------------
        # "Inclusive" mask = this layer + every layer closer to the camera
        # (i.e. higher depth values). Blurring it by the layer's own radius
        # and multiplying by the layer's blurred alpha gives us a matte that
        # is HARSH against closer layers (because the inclusive mask stays at
        # 1 well into their territory, so the blur barely tapers there) and
        # SOFT against farther layers (because the inclusive mask drops to 0
        # there, so the blur tapers normally). Closer layers, composited on
        # top, then cover their side cleanly with no bleed.
        inclusive_mask = (depth >= lo).astype(np.float32)
        cv2.imwrite(
            os.path.join(TEMP_DIR, f"{i}_inclusive.png"),
            (inclusive_mask * 255.0).astype(np.uint8),
        )

        blurred_inclusive = lens_blur(inclusive_mask, radius)
        adjusted_alpha = blurred_inclusive
        cv2.imwrite(
            os.path.join(TEMP_DIR, f"{i}_adjusted_alpha.png"),
            np.clip(adjusted_alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        )

        # ---- Composite onto the canvas (Normal blend) ----------------------
        a = np.clip(adjusted_alpha, 0.0, 1.0)[..., None]
        canvas = unmult * a + canvas * (1.0 - a)

        tag = " (in focus)" if in_focus else ""
        print(
            f"  slice {i + 1}/{DEPTH_COUNT}: depth in [{lo:.0f}, {hi:.0f}) "
            f"-> blur radius {radius}px, highlight {highlight_strength:.2f}{tag}"
        )

    # Undo the divergence warp. This is exact in the analytic sense (g is the
    # algebraic inverse of f); the only loss is one bilinear resample. Then
    # crop the padding back off to recover the original frame.
    if SWIRL_AMOUNT > 0.0:
        print(f"Converging composite (a = {SWIRL_AMOUNT}) ...")
        canvas = converge(canvas, SWIRL_AMOUNT)
        canvas = canvas[pad_y:pad_y + orig_h, pad_x:pad_x + orig_w]

    cv2.imwrite(
        os.path.join(TEMP_DIR, "final.png"),
        np.clip(canvas, 0.0, 255.0).astype(np.uint8),
    )

    print(f"Done. Wrote {DEPTH_COUNT * 6 + 2} files to {TEMP_DIR}")


if __name__ == "__main__":
    main()
