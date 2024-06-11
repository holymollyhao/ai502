import os
import json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from util import horizontal_flip, vertical_flip, scale_down, scale_up, rotate, center_crop, draw_circle, adjust_color, zoom_image, no_change
import requests

relation_to_subcategory = {
    # Adjacency
    "adjacent to": "adjacency",
    "alongside": "adjacency",
    "at the side of": "adjacency",
    "at the right side of": "adjacency",
    "at the left side of": "adjacency",
    "attached to": "adjacency",
    "at the back of": "adjacency",
    "ahead of": "adjacency",
    "against": "adjacency",
    "at the edge of": "adjacency",
    # Directional
    "off": "directional",
    "past": "directional",
    "toward": "directional",
    "down": "directional",
    "away from": "directional",
    "along": "directional",
    "around": "directional",
    "into": "directional",
    "across": "directional",
    "across from": "directional",
    "down from": "directional",
    # Orientation
    "facing": "orientation",
    "facing away from": "orientation",
    "parallel to": "orientation",
    "perpendicular to": "orientation",
    # Proximity
    "by": "proximity",
    "close to": "proximity",
    "near": "proximity",
    "far from": "proximity",
    "far away from": "proximity",
    # Topological
    "connected to": "topological",
    "detached from": "topological",
    "has as a part": "topological",
    "part of": "topological",
    "contains": "topological",
    "within": "topological",
    "at": "topological",
    "on": "topological",
    "in": "topological",
    "with": "topological",
    "surrounding": "topological",
    "among": "topological",
    "consists of": "topological",
    "out of": "topological",
    "between": "topological",
    "inside": "topological",
    "outside": "topological",
    "touching": "topological",
    # Unallocated
    "beyond": "directional",
    "next to": "proximity",
    "opposite to": "directional",
    "enclosed by": "topological",
    "above": "directional",
    "below": "directional",
    "behind": "directional",
    "on top of": "directional",
    "under": "directional",
    "over": "directional",
    "left of": "orientation",
    "right of": "orientation",
    "in front of": "directional",
    "beneath": "directional",
    "beside": "proximity",
    "in the middle of": "topological",
    "congruent": "topological",
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
    "on top of": "below of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
    "not beneath": "beneath",
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


# generated via GPT-3
relation_aware_relation_set = {
    'at the right side of': 'at the left side of',
    'at the left side of': 'at the right side of',
    'left of': 'right of',
    'right of ': 'left of',
    'near': 'near',
    'close to': 'close to',
    'far from': 'far from',
    'above': 'below',
    'below': 'above',
    'on top of': 'below',
    'down from': 'up from',
    'under': 'over',
    'over': 'under',
    'beneath': 'not beneath',
    'facing away from': 'facing away from',
    'parallel to': 'parallel to',
    'perpendicular to': 'perpendicular to',
}


relation_aware_augmentation = {
    'at the right side of': horizontal_flip,
    'at the left side of': horizontal_flip,
    'left of': horizontal_flip,
    'right of ': horizontal_flip,
    'near': scale_down,
    'close to': scale_down,
    'far from': scale_up,
    'above': vertical_flip,
    'below': vertical_flip,
    'on top of': vertical_flip,
    'down from': vertical_flip,
    'under': vertical_flip,
    'over': vertical_flip,
    'beneath': vertical_flip,
    'facing away from': horizontal_flip,
    'parallel to': horizontal_flip,
    'perpendicular to': horizontal_flip,

}


class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, vilt_processor=None, filter_relations=None, flips=[], negation=False,
                 visual_prompt=False, visual_json_path=None, synonym_objects=False, remaining_flip_handler=None):
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
        self.synonym_objects = synonym_objects
        self.remaining_flip_handler = remaining_flip_handler

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
        return bbox

    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        relation = data_point["relation"]

        try:
            vertical_relation = negate_vertical_flip[relation]
            horizontal_relation = negate_horizontal_flip[relation]
            relation_aware_relation = relation_aware_relation_set[relation]
            relation_aware_flip = relation_aware_augmentation[relation]

        except KeyError:
            vertical_relation = relation
            horizontal_relation = relation
            relation_aware_relation = relation
            if self.remaining_flip_handler == 'zoom':
                relation_aware_flip = zoom_image
            elif self.remaining_flip_handler == 'adjust_color':
                relation_aware_flip = adjust_color
            else:
                relation_aware_flip = no_change
        negative_relation = negate[relation]
        negative_relation_aware_relation = negate[relation_aware_relation]

        full_subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
        full_subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
        full_subject_a = full_subject_a.strip(".").lower()
        full_subject_b = full_subject_b.strip(".").lower()
        stopwords = []
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
            relation_aware_relation = modifier[0] + " " + relation_aware_relation
        else:
            pass
        full_subject_a = " ".join(full_subject_a)
        full_subject_b = " ".join(full_subject_b)

        if self.synonym_objects:
            full_subject_a = self.get_synonym_objects(full_subject_a).join(', ')
            full_subject_b = self.get_synonym_objects(full_subject_b).join(', ')

        if self.visual_prompt:
            full_subject_a = 'red circle around ' + full_subject_a
            full_subject_b = 'blue circle around ' + full_subject_b



        # contains
        stopwords = ['the', 'is']
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
        if 'relation_aware_flip' in self.flips:
            # if relation_aware_relation == relation:
            captions.extend(
                [
                    # false case first
                    full_subject_a + ' "' + negative_relation_aware_relation + '" ' + full_subject_b,
                    full_subject_a + ' "' + relation_aware_relation + '" ' + full_subject_b,
                ])

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

        if 'relation_aware_flip' in self.flips:
            images.append(relation_aware_flip(image))

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
