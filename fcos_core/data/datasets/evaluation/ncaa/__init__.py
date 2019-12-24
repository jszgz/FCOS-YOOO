import logging

from .ncaa_eval import do_ncaa_evaluation


def ncaa_evaluation(dataset, predictions, output_folder, predictions_event):
    logger = logging.getLogger("fcos_core.inference")

    return do_ncaa_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        predictions_event = predictions_event
    )
