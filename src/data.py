
import os
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from torch.utils.data import Dataset
from util import horizontal_flip, vertical_flip, scale_down, scale_up
spatial_relation = ['at the right side of', 'at the left side of', 'left of', 'right of', 'above', 'below']
negate = {
    # Adjacency
    "adjacent to": "nonadjacent to",
    "alongside": "away from",
    "at the side of": "away from",
    "at the right side of": "at the left side of",
    "at the left side of": "at the right side of",
    "attached to": "disconnect from",
    "at the back of": "at the front of",
    "ahead of": "behind",
    "against": "away from",
    "at the edge of": "far from the edge of",
    # Directional
    "off": "on",
    "past": "before",
    "toward": "away from",
    "down": "up",
    "away from": "not away from",
    "along": "not along",
    "around": "not around",
    "into": "not into",
    "across": "not accross",
    "across from": "not across from",
    "down from": "up from",
    # Orientation
    "facing": "facing away from",
    "facing away from": "facing",
    "parallel to": "perpendicular to",
    "perpendicular to": "parallel to",
    # Proximity
    "by": "far away from",
    "close to": "far from",
    "near": "far from",
    "far from": "close to",
    "far away from": "by",
    # Topological
    "connected to": "detached from",
    "detached from": "connected to",
    "has as a part": "does not have a part",
    "part of": "not part of",
    "contains": "does not contain",
    "within": "outside",
    "at": "not at",
    "on": "not on",
    "in": "not in",
    "with": "not with",
    "surrounding": "not surrounding",
    "among": "not among",
    "consists of": "does not consists of",
    "out of": "not out of",
    "between": "not between",
    "inside": "outside",
    "outside": "inside",
    "touching": "not touching",
    # Unallocated
    "beyond": "inside",
    "next to": "far from",
    "opposite to": "same as",
    "enclosed by": "not enclosed by",
    # missing
    "above": "below",
    "below": "above",
    "behind": "infront",
    "on top of": "not on top of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
}

negate_horizontal_flip = {
    'at the right side of': 'at the left side of', 
    'at the left side of': 'at the right side of',
    'left of': 'right of',
    'right of ': 'left of',
    'near': 'near',
    'close to': 'close to',
    'far from': 'far from',
    'above': 'above',
    'below': 'below'
}

negate_vertical_flip = {
    'at the right side of': 'at the right side of', 
    'at the left side of': 'at the left side of',
    'left of': 'left of',
    'right of ': 'right of',
    'near': 'near',
    'close to': 'close to',
    'far from': 'far from',
    'above': 'below',
    'below': 'above'
}

adaptive_relation_set = {
    'at the right side of': 'at the left side of',
    'at the left side of': 'at the right side of',
    'left of': 'right of',
    'right of ': 'left of',
    'near': 'near',
    'close to': 'close to',
    'far from': 'far from',
    'above': 'below',
    'below': 'above'
}

adaptive_augmentation ={
    'at the right side of': horizontal_flip,
    'at the left side of': horizontal_flip,
    'left of': horizontal_flip,
    'right of ': horizontal_flip,
    'near': scale_down,
    'close to': scale_down,
    'far from': scale_up,
    'above': vertical_flip,
    'below': vertical_flip
}

synonym_relation_set = {
    'at the right side of': ['right beside', 'on the right of', 'to the right of'],
    'at the left side of': ['left beside', 'on the left of', 'to the left of'],
    'left of': ['on the left', 'to the left of', 'leftward of'],
    'right of': ['on the right', 'to the right of', 'rightward of'],
    'near': ['close to', 'nearby', 'adjacent to'],
    'close to': ['near', 'next to', 'alongside'],
    'far from': ['distant from', 'away from', 'not near'],
    'above': ['over', 'on top of', 'higher than'],
    'below': ['under', 'beneath', 'lower than']
}


# load image in real time version
class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, vilt_processor=None, filter_relations=None, flips=[]): 
        self.img_path = img_path

        self.imgs = {}

        self.data_json = []
        self.flips = flips
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                if filter_relations != None and j_line["relation"] not in filter_relations:
                    # do not add!
                    continue
                self.data_json.append(j_line)
    
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        relation = data_point["relation"]
        vertical_relation = negate_vertical_flip[relation]
        horizontal_relation = negate_horizontal_flip[relation]
        negative_relation = negate[relation]
        adaptive_relation = adaptive_relation_set[relation]
        adaptive_flip = adaptive_augmentation[relation]

        full_subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
        full_subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
        full_subject_a = full_subject_a.strip(".").lower()
        full_subject_b = full_subject_b.strip(".").lower()
        # stopwords = ['the']
        stopwords = []
        # remove stop words
        full_subject_a  = [word for word in full_subject_a.split() if word.lower() not in stopwords]
        full_subject_b  = [word for word in full_subject_b.split() if word.lower() not in stopwords]


        # move modifier to the predicate
        modifier = [word for word in full_subject_a if word in ['is']]
        if len(modifier) > 0:
            full_subject_a.remove(modifier[0])
            relation = modifier[0] + " " + relation
            vertical_relation = modifier[0] + " " + vertical_relation
            horizontal_relation = modifier[0] + " " + horizontal_relation
            negative_relation = modifier[0] + " " + negative_relation
            adaptive_relation = modifier[0] + " " + adaptive_relation
        else:
            # print ('relation:', relation, ':', data_point["caption"])
            pass
        # convert word array to string
        full_subject_a = " ".join(full_subject_a)
        full_subject_b = " ".join(full_subject_b)

        # contains
        stopwords = ['the', 'is']

        if 'adaptive_flip' in self.flips:
            captions = [
                # false case first
                full_subject_a + ' ' + negative_relation + ' ' +  full_subject_b,
                full_subject_a + ' ' +  relation + ' ' +  full_subject_b,
            ]
        else:
            captions = [
                # false case first
                full_subject_a + ' ' +  relation,
                full_subject_a + ' ' +  relation + ' ' +  full_subject_b,
            ]

        if 'vertical_flip' in self.flips:
            captions.extend(
                [
                # false case first
                full_subject_a + ' ' +  vertical_relation,
                full_subject_a + ' ' +  vertical_relation + ' ' +  full_subject_b,
                ]
            )

        if 'horizontal_flip' in self.flips:
            captions.extend(
                [
                # false case first
                full_subject_a + ' ' +  horizontal_relation,
                full_subject_a + ' ' +  horizontal_relation + ' ' +  full_subject_b,
                ]
            )

        if 'adaptive_flip' in self.flips:
            captions.extend(
                [
                # false case first
                full_subject_a + ' ' +  relation + ' ' + full_subject_b,
                full_subject_a + ' ' +  adaptive_relation + ' ' +  full_subject_b,
                ]
            )

        # load Image
        img_path = os.path.join(self.img_path, data_point["image"])
        image = Image.open(img_path)

        images = [
            image,
        ]
        if 'vertical_flip' in self.flips:
            images.append(vertical_flip(image))

        if 'horizontal_flip' in self.flips:
            images.append(horizontal_flip(image))

        if 'adaptive_flip' in self.flips:
            images.append(adaptive_flip(image))

        return images, captions, data_point["label"], data_point["image"],  data_point["relation"]

    def __len__(self):
        return len(self.data_json)

    # def __getitem__(self, idx):
    #     data_point = self.data_json[idx]
    #     stopwords = ['the', 'is']
    #     relation = data_point["relation"]
    #     adaptive_relation = adaptive_relation_set[relation]
    #     adaptive_flip = adaptive_augmentation[relation]
    #
    #
    #     subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
    #     subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
    #     subject_a = [word for word in subject_a.split() if word.lower() not in stopwords][0].strip(".")
    #     subject_b = [word for word in subject_b.split() if word.lower() not in stopwords][0].strip(".")
    #
    #
    #     used_relation = []
    #     # use negative dict
    #     captions = self.ret_caption(subject_a, subject_b, relation, negate[relation])
    #     used_relation.append(relation)
    #
    #
    #     if 'adaptive_flip' in self.flips:
    #         captions.extend(self.ret_caption(subject_a, subject_b, relation, adaptive_relation))
    #         used_relation.append(adaptive_relation_set[relation])
    #
    #     if 'synonym' in self.flips:
    #         updated_captions = []
    #         for idx, rel in enumerate(used_relation):
    #             synonm_true = synonym_relation_set[rel]
    #             synonm_false = synonym_relation_set[negate[rel]]
    #             appending_caption_list = captions[2*idx : 2*idx+2]
    #             for true_rel, false_rel in zip(synonm_true, synonm_false):
    #                 appending_caption_list.extend(self.ret_caption(subject_a, subject_b, false_rel, true_rel))
    #             updated_captions.extend(appending_caption_list)
    #         captions = updated_captions
    #
    #     # load Image
    #     img_path = os.path.join(self.img_path, data_point["image"])
    #     # image = cv2.imread(img_path)
    #     images = []
    #     images.append(Image.open(img_path))
    #
    #     if 'adaptive_flip' in self.flips:
    #         images.append(adaptive_flip(Image.open(img_path)))
    #
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(images[0])
    #     # plt.show()
    #
    #     # plt.imshow(images[1])
    #     # plt.show()
    #
    #     return images, captions, data_point["label"], data_point["image"], data_point["relation"]
    #     # return self.imgs[data_point["image"]], captions, data_point["label"]
    #     # return self.imgs[data_point["image"]], data_point["caption"], data_point["label"]
    # 

    def ret_caption(self, obj1, obj2, rel1, rel2):
        if rel2 in spatial_relation:
            ret_list = [
                # false case first
                obj2 + ' ' + rel2 ,

                # ture case second
                obj1 + ' ' + rel2 + ' ' + obj2,
            ]
        else:
            ret_list = [
                # false case first
                obj1 + ' ' + rel1 ,

                # ture case second
                obj1 + ' ' + rel2 + ' ' + obj2,
            ]
        return ret_list

