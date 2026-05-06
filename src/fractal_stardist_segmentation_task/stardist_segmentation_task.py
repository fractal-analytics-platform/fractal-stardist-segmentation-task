"""StarDist segmentation task for Fractal."""

import logging

from fractal_tasks_utils.segmentation import (
    IteratorConfig,
    compute_segmentation,
    setup_segmentation_iterator,
)
from fractal_tasks_utils.segmentation._transforms import SegmentationTransformConfig
from ngio import ChannelSelectionModel, open_ome_zarr_container
from pydantic import Field, validate_call

from fractal_stardist_segmentation_task.segmentation_utils import (
    load_stardist_model,
    segment_image,
)
from fractal_stardist_segmentation_task.utils import (
    AnyCreateRoiTableModel,
    CreateMaskingRoiTable,
    SkipCreateMaskingRoiTable,
)

logger = logging.getLogger("stardist_segmentation_task")


@validate_call
def stardist_segmentation_task(
    *,
    # Fractal managed parameters
    zarr_url: str,
    # Segmentation parameters
    channel: ChannelSelectionModel,
    label_name: str = "{channel_identifier}_stardist_segmented",
    level_path: str | None = None,
    # StarDist model parameters
    stardist_model: str = "2D_versatile_fluo",
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
    # Iterator / infrastructure parameters
    iterator_configuration: IteratorConfig | None = None,
    pre_post_process: SegmentationTransformConfig = Field(  # noqa: B008
        default_factory=SegmentationTransformConfig
    ),
    create_masking_roi_table: AnyCreateRoiTableModel = Field(  # noqa: B008
        default_factory=SkipCreateMaskingRoiTable
    ),
    overwrite: bool = True,
) -> None:
    """Segment an image using StarDist.

    Runs StarDist instance segmentation on a Fractal OME-Zarr dataset.
    Supports both 2D and 3D pretrained models. The model is selected by name
    and downloaded/cached automatically on first use.

    Args:
        zarr_url (str): URL to the OME-Zarr container.
        channel (ChannelSelectionModel): Select the input channel to be used
            for segmentation.
        label_name (str): Name of the resulting label image. Can contain a
            placeholder "{channel_identifier}" which will be replaced by the
            channel identifier specified in the channel parameter.
        level_path (str | None): If the OME-Zarr has multiple resolution
            levels, the level to use can be specified here. If not provided,
            the highest resolution level will be used.
        stardist_model (str): Name of the pretrained StarDist model to use.
            2D models: "2D_versatile_fluo", "2D_versatile_he",
            "2D_paper_dsb2018". 3D models: "3D_demo".
            Defaults to "2D_versatile_fluo".
        prob_thresh (float | None): Probability threshold for instance
            detection. If None, the model's default threshold is used.
        nms_thresh (float | None): Non-maximum suppression threshold for
            instance overlap removal. If None, the model's default is used.
        iterator_configuration (IteratorConfig | None): Advanced configuration
            to control masked and ROI-based iteration.
        pre_post_process (SegmentationTransformConfig): Configuration for pre-
            and post-processing transforms applied by the iterator.
        create_masking_roi_table (AnyCreateRoiTableModel): Configuration to
            create a masking ROI table after segmentation.
        overwrite (bool): Whether to overwrite an existing label image.
            Defaults to True.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr container
    ome_zarr = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    # Format the label name based on the provided template and channel identifier
    label_name = label_name.format(channel_identifier=channel.identifier)
    logger.info(f"Formatted label name: {label_name=}")

    # Load the StarDist model (with retry logic for cluster race conditions)
    model = load_stardist_model(stardist_model=stardist_model)

    # Set up the segmentation iterator
    iterator = setup_segmentation_iterator(
        zarr_url=zarr_url,
        channels=[channel],
        output_label_name=label_name,
        level_path=level_path,
        iterator_configuration=iterator_configuration,
        segmentation_transform_config=pre_post_process,
        overwrite=overwrite,
    )

    # Run the core segmentation loop
    compute_segmentation(
        segmentation_func=lambda x: segment_image(
            image=x,
            model=model,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        ),
        iterator=iterator,
    )
    logger.info(f"label {label_name} successfully created at {zarr_url}")

    # Build a masking ROI table if configured
    if isinstance(create_masking_roi_table, CreateMaskingRoiTable):
        table_name = create_masking_roi_table.get_table_name(label_name=label_name)
        label = ome_zarr.get_label(name=label_name, path=level_path)
        masking_roi_table = label.build_masking_roi_table()
        ome_zarr.add_table(
            name=table_name, table=masking_roi_table, overwrite=overwrite
        )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stardist_segmentation_task)
