
import os
import json
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from torch.utils.data import Dataset

# load image in real time version
class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, vilt_processor=None): 
        self.img_path = img_path

        self.imgs = {}

        self.data_json = []
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data_json.append(j_line)
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        relation = data_point["relation"]
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
        else:
            # print ('relation:', relation, ':', data_point["caption"])
            pass
        # convert word array to string
        full_subject_a = " ".join(full_subject_a)
        full_subject_b = " ".join(full_subject_b)

        # contains
        stopwords = ['the', 'is']
        captions = [
            # false case first
            full_subject_a + ' "' + relation + '"',
            full_subject_a + ' "' + relation + '" ' + full_subject_b,
        ]


        # load Image
        img_path = os.path.join(self.img_path, data_point["image"])
        image = Image.open(img_path)
        return image, captions, data_point["label"], data_point["image"]

    def __len__(self):
        return len(self.data_json)
