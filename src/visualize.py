from data import ImageTextClassificationDataset, negate_horizontal_flip, relation_to_subcategory
from transformers import AutoProcessor, AutoModel
import torch
import argparse
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
import matplotlib.colors as mcolors

from captum.attr import visualization


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    images = images.cuda()
    texts = texts.cuda()
    out = model(pixel_values=images, input_ids=texts, output_attentions=True)
    logits_per_image = out.logits_per_image
    logits_per_text = out.logits_per_text
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    # image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    # https://gist.github.com/calpt/8e3555bd11f1916b5169c8125117e5ee
    image_attn_blocks = list(dict(model.vision_model.encoder.layers.named_children()).values())

    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1
    num_tokens = out.vision_model_output.attentions[0].shape[-1]
    # num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=out.vision_model_output.attentions[0].dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        attn_probs = out.vision_model_output.attentions[i]
        grad = torch.autograd.grad(one_hot, [attn_probs], retain_graph=True)[0].detach()
        cam = attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    # text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())
    text_attn_blocks = list(dict(model.text_model.encoder.layers.named_children()).values())

    if start_layer_text == -1:
        # calculate index of last layer
        start_layer_text = len(text_attn_blocks) - 1

    # num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    num_tokens = out.text_model_output.attentions[0].shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=out.text_model_output.attentions[0].dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        attn_probs = out.text_model_output.attentions[i]
        grad = torch.autograd.grad(one_hot, [attn_probs], retain_graph=True)[0].detach()
        cam = attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance


def show_image_relevance(image_relevance, image, orig_image, size=224):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=size, mode='bilinear')
    image_relevance = image_relevance.reshape(size, size).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');


def get_color(score):
    cmap = plt.get_cmap('viridis')
    return cmap(score)
    # red = max(0, score)
    # blue = -min(0, score)
    # return (red, 0, blue)


def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    # R_text = R_text[CLS_idx, :]
    #   print(R_text.shape, CLS_idx)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    # print(text_scores)
    text_tokens = _tokenizer.encode(text)
    # print(len(text_tokens))
    # print(text_encoding.shape)
    # print(text_scores)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens[1:CLS_idx]]
    x = 0.0
    # plt.figure(figsize=(12, 2))
    plt.axis('off')
    text_scores = text_scores.softmax(dim=-1)
    # print(torch.sum(text_scores))
    # text_scores = torch.softmax(text_scores)
    text_scores = (text_scores - torch.min(text_scores)) / (torch.max(text_scores) - torch.min(text_scores))
    for token, score in zip(text_tokens_decoded, text_scores):
        color = get_color(score=score.item())
        # print(token, color, len(token))
        x += 0.02 * len(token) / 2  # Adjust spacing based on token length
        # print(x)
        plt.text(x, 0.5, token, fontsize=12, color=color, ha='center', va='center')
        x += 0.02 * len(token) / 2
        x += 0.01
    # plt.tight_layout()


#   text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
#   vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
#   visualization.visualize_text(vis_data_records)

def visualize(data_loader, model):
    model.cuda()
    model.eval()
    for i, data in tqdm(enumerate(data_loader)):
        input_ids, pixel_values, y, captions, imgs, category = data
        category = category[0]
        y = y.cuda()
        R_text, R_image = interpret(model=model, image=pixel_values, texts=input_ids, device='cuda')
        batch_size = input_ids.shape[0]
        # print(captions)
        # print(imgs[0])
        plt.figure()
        for j in range(batch_size):
            item1 = 'bird' in captions[j] and 'cat' in captions[j] and 'above' in captions[j]
            item2 = 'bed' in captions[j] and 'bench' in captions[j] and 'left' in captions[j]
            if item1 or item2:
                show_heatmap_on_text(captions[j], input_ids[j], R_text[j])
                plt.savefig(f"{result_path}/{i}_text.jpg")
                show_image_relevance(R_image[j], pixel_values, orig_image=imgs[0][0], size=336)
                plt.savefig(f"{result_path}/{i}_image.jpg")
                break

def map_keys(source_dict):
    target_dict = {}
    for key, value in source_dict.items():
        if key.startswith("module."):
            target_dict[key[7:]] = value
    return target_dict

result_path = "results"
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize')
    parser.add_argument('--model_path', type=str, default="openai/clip-vit-large-patch14-336")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.load_state_dict(map_keys(torch.load("clip_adaptive_tune_wnone_textencoder/best_checkpoint/ckpt.pt")))
    _tokenizer = processor.tokenizer

    os.makedirs(result_path, exist_ok=True)


    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames, category = zip(*batch)
        # print(captions[0][1:2])
        inputs = processor(captions[0][1:2], images=imgs[0], return_tensors="pt")
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values

        labels = torch.tensor(labels)
        return input_ids, pixel_values, labels, list(captions[0][1:2]), imgs, category


    collate_fn_batch = collate_fn_batch_clip
    batch_size = 1
    json_path = os.path.join('../data', 'splits', 'random', 'train.jsonl')  # 10119 image text pairs
    img_path = os.path.join('../data', 'images')  # changes to images
    visual_json_path = os.path.join('../data', 'annotations', 'panoptic_train2017.json')
    dataset = ImageTextClassificationDataset(img_path, json_path, filter_relations=None, flips=[])
    # dataset = ImageTextClassificationDataset(img_path, json_path, filter_relations=None, flips=[], visual_prompt=True, visual_json_path=visual_json_path, add_arrow=True, bbox_color="blue", bbox_type="bbox", add_grid=False)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn_batch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0, )
    visualize(test_loader, model)