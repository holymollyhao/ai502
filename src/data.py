import os
import json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from util import horizontal_flip, vertical_flip, scale_down, scale_up, rotate, center_crop, draw_circle, adjust_color
import requests

relation_to_subcategory = {
    "adjacent to": "adjacency",
    "alongside": "adjacency",
    "at the side of": "adjacency",
    "attached to": "adjacency",
    "connected to": "adjacency",
    "detached from": "adjacency",
    "touching": "adjacency",
    "next to": "adjacency",
    "beside": "adjacency",

    "at the right side of": "directional",
    "at the left side of": "directional",
    "at the back of": "directional",
    "ahead of": "directional",
    "off": "directional",
    "past": "directional",
    "toward": "directional",
    "down": "directional",
    "down from": "directional",
    "facing": "directional",
    "facing away from": "directional",
    "parallel to": "directional",
    "perpendicular to": "directional",
    "by": "directional",
    "above": "directional",
    "below": "directional",
    "behind": "directional",
    "under": "directional",
    "over": "directional",
    "left of": "directional",
    "right of": "directional",
    "in front of": "directional",
    "beneath": "directional",

    "at the edge of": "orientation",
    "along": "orientation",
    "around": "orientation",
    "into": "orientation",
    "across": "orientation",
    "across from": "orientation",
    "by": "orientation",
    "between": "orientation",
    "beyond": "orientation",
    "opposite to": "orientation",
    "on top of": "orientation",

    "close to": "proximity",
    "near": "proximity",
    "far from": "proximity",
    "far away from": "proximity",
    "within": "proximity",
    "at": "proximity",
    "on": "proximity",
    "in": "proximity",
    "with": "proximity",
    "surrounding": "proximity",
    "among": "proximity",
    "out of": "proximity",
    "inside": "proximity",
    "outside": "proximity",
    "enclosed by": "proximity",

    "has as a part": "topological",
    "part of": "topological",
    "contains": "topological",
    "consists of": "topological",
    "congruent": "topological",
    "not adjacent to": "topological",
    "away from": "topological",
    "disconnect from": "topological",
    "far from the edge of": "topological",
    "not along": "topological",
    "not around": "topological",
    "not into": "topological",
    "not across": "topological",
    "up from": "topological",
    "perpendicular to": "topological",
    "far away from": "topological",
    "does not have a part": "topological",
    "not part of": "topological",
    "does not contain": "topological",
    "outside": "topological",
    "not at": "topological",
    "not on": "topological",
    "not in": "topological",
    "not with": "topological",
    "not surrounding": "topological",
    "not among": "topological",
    "does not consists of": "topological",
    "not out of": "topological",
    "not between": "topological",
    "inside": "topological",
    "not touching": "topological",
    "not enclosed by": "topological",
    "incongruent": "topological",
}

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

adaptive_augmentation = {
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


# load image in real time version
class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, vilt_processor=None, filter_relations=None, flips=[], negation=False,
                 visual_prompt=False, visual_json_path=None, synonym_objects=False):
        self.img_path = img_path

        self.imgs = {}
        self.negation = negation

        self.data_json = []
        self.flips = flips
        self.visual_prompt = visual_prompt
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                if filter_relations != None and j_line["relation"] not in filter_relations:
                    # do not add!
                    continue
                self.data_json.append(j_line)
        if self.visual_prompt:
            self.id_category_map = dict()
            self.image_name_to_data = dict()
            with open(visual_json_path, "r") as f:
                annot_data = json.load(f)
            id_to_image_path = dict()
            for img in annot_data['images']:
                id_to_image_path[img['id']] = img['file_name']
            for annot in annot_data['annotations']:
                self.image_name_to_data[id_to_image_path[annot['image_id']]] = annot['segments_info']
            for cat in annot_data['categories']:
                self.id_category_map[cat['id']] = cat['name']
            # print(self.id_category_map)

            with open(visual_json_path.replace("train", "val"), "r") as f:
                annot_data = json.load(f)
            id_to_image_path = dict()
            for img in annot_data['images']:
                id_to_image_path[img['id']] = img['file_name']
            for annot in annot_data['annotations']:
                self.image_name_to_data[id_to_image_path[annot['image_id']]] = annot['segments_info']
            for cat in annot_data['categories']:
                self.id_category_map[cat['id']] = cat['name']
            # print(self.id_category_map)
        self.synonym_objects = synonym_objects

    def get_bbox(self, data_point):
        if 'subj' in data_point:
            names = [data_point['subj'], data_point['obj']]
        else:
            tmp_names = data_point['caption'].split(f"is {data_point['relation']}")
            names = [name[4:-1].strip() for name in tmp_names]
        target_img = data_point['image']
        if target_img not in self.image_name_to_data:
            bbox = [None, None]
            print("WARNING", target_img)
        else:
            segments_info = self.image_name_to_data[target_img]
            bbox = [None, None]
            for seg in segments_info:
                if self.id_category_map[seg['category_id']] == names[0]:
                    x, y, w, h = seg['bbox']
                    bbox[0] = [x, y, x + w, y + h]
                if self.id_category_map[seg['category_id']] == names[1]:
                    x, y, w, h = seg['bbox']
                    bbox[1] = [x, y, x + w, y + h]
            # if None in bbox:
            #     print(bbox, names)
            #     import pdb;pdb.set_trace()
        return bbox

    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        relation = data_point["relation"]

        try:
            vertical_relation = negate_vertical_flip[relation]
            horizontal_relation = negate_horizontal_flip[relation]
            adaptive_relation = adaptive_relation_set[relation]
            adaptive_flip = adaptive_augmentation[relation]

        except KeyError:
            vertical_relation = relation
            horizontal_relation = relation
            adaptive_relation = relation
            adaptive_flip = adjust_color

        negative_relation = negate[relation]
        negative_adaptive_relation = negate[adaptive_relation]

        full_subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
        full_subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
        full_subject_a = full_subject_a.strip(".").lower()
        full_subject_b = full_subject_b.strip(".").lower()
        # stopwords = ['the']
        stopwords = []
        # remove stop words
        full_subject_a = [word for word in full_subject_a.split() if word.lower() not in stopwords]
        full_subject_b = [word for word in full_subject_b.split() if word.lower() not in stopwords]

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

        if self.synonym_objects:
            full_subject_a = self.get_synonym_objects(full_subject_a).join(', ')
            full_subject_b = self.get_synonym_objects(full_subject_b).join(', ')

        if self.visual_prompt:
            # full_subject_a = full_subject_a + ' in red circle'
            # full_subject_b = full_subject_b + ' in blue circle'

            full_subject_a = 'red circle around ' + full_subject_a
            full_subject_b = 'blue circle around ' + full_subject_b

            # full_subject_a = 'red circle'
            # full_subject_b = 'blue circle'



        # contains
        stopwords = ['the', 'is']

        # if 'adaptive_flip' in self.flips:
        #     captions = [
        #         # false case first
        #         full_subject_a + ' "' + negative_relation + '"' + full_subject_b,
        #         full_subject_a + ' "' + relation + '" ' + full_subject_b,
        #     ]
        # else:
        captions = [
            # false case first
            full_subject_a + ' "' + negative_relation + '" ' + full_subject_b,
            full_subject_a + ' "' + relation + '" ' + full_subject_b,
        ]

        if 'rotate' in self.flips or 'center_crop' in self.flips:
            for _ in range(2):
                captions.extend([
                    # false case first
                    full_subject_a + ' "' + relation + '"',
                    full_subject_a + ' "' + relation + '" ' + full_subject_b,
                ])
        if 'vertical_flip' in self.flips:
            captions.extend(
                [
                    # false case first
                    full_subject_a + ' "' + vertical_relation + '"',
                    full_subject_a + ' "' + vertical_relation + '" ' + full_subject_b,
                ]
            )

        if 'horizontal_flip' in self.flips:
            captions.extend(
                [
                    # false case first
                    full_subject_a + ' "' + horizontal_relation + '"',
                    full_subject_a + ' "' + horizontal_relation + '" ' + full_subject_b,
                ]
            )
        #
        if 'adaptive_flip' in self.flips:
            # if adaptive_relation == relation:
            captions.extend(
                [
                    # false case first
                    full_subject_a + ' "' + negative_adaptive_relation + '" ' + full_subject_b,
                    full_subject_a + ' "' + adaptive_relation + '" ' + full_subject_b,
                ]
            )
            # else:
            #     captions.extend(
            #         [
            #             # false case first
            #             full_subject_a + ' "' + adaptive_relation + '" ' + full_subject_b,
            #             full_subject_a + ' "' + negative_adaptive_relation + '" ' + full_subject_b,
            #         ]
            #     )

        # load Image
        img_path = os.path.join(self.img_path, data_point["image"])
        image = Image.open(img_path)

        if self.visual_prompt:
            bbox = self.get_bbox(data_point=data_point)
            if bbox[0] != None:
                image = draw_circle(image=image, bbox=bbox[0], color="red")
            else:
                print("No red")
            if bbox[1] != None:
                image = draw_circle(image=image, bbox=bbox[1], color="blue")
            else:
                print("No blue")

        images = [
            image,
        ]

        if 'vertical_flip' in self.flips:
            images.append(vertical_flip(image))

        if 'horizontal_flip' in self.flips:
            images.append(horizontal_flip(image))

        if 'adaptive_flip' in self.flips:
            images.append(adaptive_flip(image))

        if 'rotate' in self.flips:
            images.append(rotate(image, degree=-15))
            images.append(rotate(image, degree=15))

        if 'center_crop' in self.flips:
            images.append(center_crop(image, crop_factor=0.5))
            images.append(center_crop(image, crop_factor=0.8))

        return images, captions, data_point["label"], data_point["image"], data_point["relation"]

    def __len__(self):
        return len(self.data_json)

    def get_synonym_objects(self, object, topn=3):
        import gensim.downloader as api
        model = api.load('word2vec-google-news-300')
        print(object)
        if object in model:
            return [object] + [synonym for synonym, _ in model.most_similar(object, topn=topn)]
        else:
            return [object]
