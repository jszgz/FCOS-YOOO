
# Author: chwangteng

import os
import torch
import torch.utils.data
import pickle
from PIL import Image
#chwangteng
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fcos_core.structures.bounding_box import BoxList

import numpy as np

class NCAABasketballDataset(torch.utils.data.Dataset):

    # Explaination to one record in my annotation.
    #
    # record:dictionary =
    # {
    #     "ImagePath":string
    #     "YoutubeId":string
    #     "FrameTime":int in us(microsecond)/1000000
    #     "Boxes":list = [
    #                 [TopLeftX, TopLeftY, Width, Height, PlayerId]:list,   percentage, not absolute size
    #                 [TopLeftX, TopLeftY, Width:float, Height:float, PlayerId:string],
    #                 ...
    #                 [TopLeftX:float, TopLeftY:float, Width, Height, PlayerId],
    #             ]
    #     "Event":dictionary = {
    #         'VideoWidth': int,
    #         'VideoHeight': int,
    #         'ClipStartTime':float,                                  time in ms(millisecond)/1000
    #         'ClipEndTime': float,                                   time in ms(millisecond)/1000
    #         'EventStartTime':float ,      -1 if no event else float time in ms(millisecond)/1000
    #         'EventEndTime': float,        -1 if no event else float time in ms(millisecond)/1000
    #         'EventStartBallX':float ,     -1 if no event else float (not used in this paper)
    #         'EventStartBallY':float ,     -1 if no event else float (not used in this paper)
    #         'EventLabel':string ,         NOEVENT if no event else label
    #         'TrainValOrTest':string       train,test,val
    #     }
    # }

    OBJECT_CLASSES = ( # 2 classes, no ball detection annotation in 5fps in the dataset
        "__background__ ",
        "person",
    )

    EVENT_CLASSES =( # 11 classes, excluding steal success Event, including
        "NOEVENT",
        "3-pointer success",
        "3-pointer failure",
        "free-throw success",
        "free-throw failure",
        "layup success",
        "layup failure",
        "other 2-pointer success",
        "other 2-pointer failure",
        "slam dunk success",
        "slam dunk failure",
        #"steal success",
        # I don't use it
        # because this event has same EventStartTime and EventEndTime,
        # that is,
        # EvetStartTime is marked as -1 in the annotatiaon csv
    )

    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.ann_file = ann_file
        self.transforms = transforms

        obj_cls = NCAABasketballDataset.OBJECT_CLASSES
        self.obj_class_to_ind = dict(zip(obj_cls, range(len(obj_cls))))
        event_cls = NCAABasketballDataset.EVENT_CLASSES
        self.event_class_to_ind = dict(zip(event_cls, range(len(event_cls))))


        #load annotation
        ann_file_pkl = open(self.ann_file, 'rb')
        self.annotation = pickle.load(ann_file_pkl) #list



    def __getitem__(self, index):

        record = self.annotation[index]
        img = Image.open(os.path.join(self.root,record['ImagePath'])).convert("RGB")


        target = self.get_groundtruth(index)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.annotation)


    def get_groundtruth(self, index):

        record = self.annotation[index]

        #boxes
        boxes_record = record['Boxes']
        temp = np.array(boxes_record)
        temp = temp[:,:4].astype(np.float32)

        width = record['Event']['VideoWidth']
        height = record['Event']['VideoHeight']
        temp[:,0] = width * temp[:,0]
        temp[:, 1] = height * temp[:, 1]
        temp[:, 2] = width * temp[:, 2]
        temp[:, 3] = height * temp[:, 3]

        boxes = temp.tolist()
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = BoxList(boxes, (width, height), mode="xywh").convert("xyxy")

        classes = [1 for i in range(len(boxes_record))]
        target.add_field("labels", torch.tensor(classes))

        # TODO: need to preprocess

        EventStartTime = record['Event']['EventStartTime']
        EventEndTime = record['Event']['EventEndTime']
        EventStartTimeOffsetInSecond = record['FrameTime']/1000000.0 - EventStartTime/1000.0
        EventEndTimeOffsetInSecond = EventEndTime/1000.0 - record['FrameTime']/1000000.0

        # events_target = {
        #     'EventLabel':torch.tensor(self.event_class_to_ind[record['Event']['EventLabel']]),
        #     'EventStartTimeOffsetInSecond':torch.tensor(EventStartTimeOffsetInSecond),
        #     'EventEndTimeOffsetInSecond': torch.tensor(EventEndTimeOffsetInSecond),
        # }
        target.add_field("EventLabel", torch.tensor(self.event_class_to_ind[record['Event']['EventLabel']]))
        target.add_field("EventStartTimeOffsetInSecond", torch.tensor(EventStartTimeOffsetInSecond))
        target.add_field("EventEndTimeOffsetInSecond", torch.tensor(EventEndTimeOffsetInSecond))
        target.add_field("FrameTime", torch.tensor(record['FrameTime']))
        #add this and I don't need to modify ncaa_eval from the copy of voc_eval
        difficult = [False for i in range(len(boxes_record))]
        target.add_field("difficult", torch.tensor(difficult))

        return target



    def get_img_info(self, index):

        record = self.annotation[index]

        height = record['Event']['VideoHeight']
        width = record['Event']['VideoWidth']
        return {"height": height, "width": width}

    def map_obj_class_id_to_class_name(self, class_id):
        return NCAABasketballDataset.OBJECT_CLASSES[class_id]

    def map_event_class_id_to_class_name(self, class_id):
        return NCAABasketballDataset.EVENT_CLASSES[class_id]