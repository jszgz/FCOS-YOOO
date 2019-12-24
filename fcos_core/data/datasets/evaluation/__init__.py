from fcos_core.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
from .ncaa import ncaa_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    args = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs
    )
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    #chwangteng
    elif isinstance(dataset, datasets.NCAABasketballDataset):
        args_event = dict(
        dataset=dataset, predictions=predictions, output_folder=output_folder, predictions_event=kwargs['predictions_event']
    )
        return ncaa_evaluation(**args_event)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
