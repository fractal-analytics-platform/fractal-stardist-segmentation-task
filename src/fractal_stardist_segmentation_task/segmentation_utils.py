"""Segmentation utils for StarDist."""

import logging
import random
import time
from typing import Optional

import numpy as np
from stardist.models import StarDist2D, StarDist3D

logger = logging.getLogger(__name__)


def load_stardist_model(
    stardist_model: str,
) -> StarDist2D | StarDist3D:
    """Load a pretrained StarDist model with retry logic for cluster safety.

    Retries up to 10 times with random sleep between attempts to handle
    race conditions when multiple workers download the model simultaneously.

    Args:
        stardist_model: Name of the pretrained StarDist model.
            2D models: "2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018".
            3D models: "3D_demo".

    Returns:
        Loaded StarDist2D or StarDist3D model.

    Raises:
        RuntimeError: If the model cannot be loaded after 10 attempts.
    """
    model = None
    model_loaded = False
    count = 0
    while not model_loaded and count < 10:
        try:
            if "3D" in stardist_model:
                model = StarDist3D.from_pretrained(stardist_model)
            else:
                model = StarDist2D.from_pretrained(stardist_model)
            if model:
                model_loaded = True
        except Exception:
            logger.warning(
                f"Failed to load StarDist model '{stardist_model}' "
                f"(attempt {count + 1}/10). Retrying..."
            )
            time.sleep(random.uniform(2, 7))
            count += 1

    if model is None:
        raise RuntimeError(
            f"Could not load StarDist model '{stardist_model}' after 10 attempts."
        )

    logger.info(f"Successfully loaded StarDist model '{stardist_model}'")
    return model


def segment_image(
    image: np.ndarray,
    model: StarDist2D | StarDist3D,
    prob_thresh: Optional[float] = None,
    nms_thresh: Optional[float] = None,
) -> np.ndarray:
    """Run StarDist instance segmentation on a single image.

    Handles both 2D and 3D images. The Fractal iterator passes channel images
    that may have extra leading singleton dimensions (e.g. shape (1, H, W) for
    a 2D image). This function strips those extra dims, runs prediction, then
    restores the original leading shape.

    Args:
        image: Input image as numpy array. May be 2D (H, W), 3D (Z, H, W), or
            have extra leading singleton dims like (1, H, W) or (1, Z, H, W).
        model: Loaded StarDist2D or StarDist3D model.
        prob_thresh: Probability threshold for instance detection. If None, the
            model's default is used.
        nms_thresh: Non-maximum suppression threshold for overlap removal.
            If None, the model's default is used.

    Returns:
        Instance segmentation label array of same shape as input, dtype uint32.
    """
    is_3d = isinstance(model, StarDist3D)
    n_spatial = 3 if is_3d else 2
    axes = "ZYX" if is_3d else "YX"

    # Strip leading singleton dimensions, remembering how many to restore
    extra_dims = image.shape[:-n_spatial]
    spatial_image = image.reshape(image.shape[-n_spatial:])

    predict_kwargs: dict = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    logger.info(
        f"Running StarDist prediction: shape={spatial_image.shape}, "
        f"axes={axes}, {predict_kwargs=}"
    )

    labels, _ = model.predict_instances(spatial_image, axes=axes, **predict_kwargs)

    logger.info(f"Generated {labels.max()} instances, shape={labels.shape}")

    # Restore leading dimensions so the ngio writer can handle them correctly
    for _ in extra_dims:
        labels = labels[np.newaxis]

    return labels.astype(np.uint32)
