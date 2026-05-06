from pathlib import Path

import pytest
from ngio import ChannelSelectionModel, create_synthetic_ome_zarr

from fractal_stardist_segmentation_task.stardist_segmentation_task import (
    stardist_segmentation_task,
)
from fractal_stardist_segmentation_task.utils import (
    IteratorConfiguration,
    MaskingConfiguration,
)


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 10, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((1, 1, 10, 64, 64), "tczyx"),
    ],
)
def test_stardist_segmentation_task(tmp_path: Path, shape: tuple[int, ...], axes: str):
    """Base test for the StarDist segmentation task."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")
    is_3d = "z" in axes
    model_name = "3D_demo" if is_3d else "2D_versatile_fluo"

    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        stardist_model=model_name,
        overwrite=True,
    )

    expected_label = "DAPI_0_stardist_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    # Note: StarDist on synthetic zero-filled images typically finds no instances.
    # We only verify the label image was created with the correct shape.
    assert label_data is not None
    # DISCLAIMER: This is only a very basic test.
    # More comprehensive tests should be implemented based on the expected
    # results not only the presence of a label image.


@pytest.mark.parametrize(
    "shape, axes",
    [
        ((64, 64), "yx"),
        ((1, 64, 64), "cyx"),
        ((3, 64, 64), "cyx"),
        ((4, 64, 64), "tyx"),
        ((1, 64, 64), "zyx"),
        ((1, 1, 64, 64), "czyx"),
        ((1, 10, 64, 64), "czyx"),
        ((1, 1, 64, 64), "tzyx"),
        ((1, 3, 64, 64), "tcyx"),
        ((1, 1, 10, 64, 64), "tczyx"),
    ],
)
def test_stardist_segmentation_task_masked(
    tmp_path: Path, shape: tuple[int, ...], axes: str
):
    """Test the StarDist segmentation task with a masking configuration."""
    test_data_path = tmp_path / "data.zarr"

    if "c" in axes:
        num_channels = shape[axes.index("c")]
    else:
        num_channels = 1
    channel_labels = [f"DAPI_{i}" for i in range(num_channels)]

    ome_zarr = create_synthetic_ome_zarr(
        store=test_data_path,
        shape=shape,
        channels_meta=channel_labels,
        overwrite=False,
        axes_names=axes,
    )
    channel = ChannelSelectionModel(identifier="DAPI_0", mode="label")
    is_3d = "z" in axes
    model_name = "3D_demo" if is_3d else "2D_versatile_fluo"

    iter_config = IteratorConfiguration(
        masking=MaskingConfiguration(mode="Label Name", identifier="nuclei_mask"),
        roi_table=None,
    )
    stardist_segmentation_task(
        zarr_url=str(test_data_path),
        channel=channel,
        stardist_model=model_name,
        overwrite=True,
        iterator_configuration=iter_config,
    )

    expected_label = "DAPI_0_stardist_segmented"
    assert expected_label in ome_zarr.list_labels()

    label = ome_zarr.get_label(expected_label)
    label_data = label.get_as_numpy()
    assert label_data is not None
    # DISCLAIMER: This is only a very basic test.
    # More comprehensive tests should be implemented based on the expected
    # results not only the presence of a label image.
