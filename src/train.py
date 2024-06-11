import os
import cv2
import json
import wandb
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim import Adam, Adadelta, Adamax, Adagrad, RMSprop, Rprop, SGD
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoConfig, BertTokenizer, VisualBertModel, \
    VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from data import ImageTextClassificationDataset, negate_horizontal_flip, relation_to_subcategory
import torch.nn.functional as F
# from eval import evaluate
# from lxmert_for_classification import LxmertForBinaryClassification

def show_image(image, caption):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    plt.title(caption)
    plt.show()


# Function to transform binary labels to unique labels for a given split
def transform_labels_for_split(binary_labels, num_texts_per_image):
    unique_labels = []
    for idx, label in enumerate(binary_labels):
        unique_label = idx % num_texts_per_image
        unique_labels.append(unique_label if label == 1 else unique_label + 1)
    return torch.tensor(unique_labels)

def evaluate(data_loader, model, flips=[], filter_relations=None):
    model.cuda()
    model.eval()

    unique_subcategories = list(set(relation_to_subcategory.values()))

    correct_orig = 0
    correct, total, all_true = 0, 0, 0
    correct_per_category = {key: 0 for key in unique_subcategories}
    total_per_category = {key: 0 for key in unique_subcategories}
    preds = []
    errors = list()
    tot_val_loss = 0
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids, pixel_values, y, captions, filenames, relation_names = data
        y = y.cuda()

        if filter_relations is not None:
            filter_idx_list = [idx for idx, relation in enumerate(relation_names) if relation in filter_relations]

        with torch.no_grad():
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            batch_cap = input_ids.cuda()
            batch_img = pixel_values.cuda()
            outputs = model(input_ids=batch_cap, pixel_values=batch_img)
        # assert len(outputs.text_embeds) == len(batch_cap)
        # reproduce huggingface webapp
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # If reshaping is necessary, ensure it aligns with your logic
        text_features = text_features.view(-1, 2, text_features.shape[-1])
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)

        score_list = []
        # print('Image and Text shape')
        # print(image_features.shape, text_features.shape)
        for j in range(len(image_features)):
            scores = image_features[j] @ text_features[j].T * 100
            score_list.append(scores)

        scores = torch.stack(score_list, dim=0).softmax(dim=-1)
        preds = scores.squeeze().argmax(-1)

        valid_loss = F.cross_entropy(scores, y.squeeze().long())
        tot_val_loss += valid_loss.item()
        sum_list = [1 if preds[i] == y[i] else 0 for i in range(len(y))]
        sum_list = [i for idx, i in enumerate(sum_list) if idx % (len(flips) + 1) == 0]
        sum_list = torch.tensor(sum_list)

        if filter_relations is not None:
            for k, (relation, correct) in enumerate(zip(relation_names, sum_list)):
                subcategory = relation_to_subcategory[relation]
                correct_per_category[subcategory] += correct
                total_per_category[subcategory] += 1

                if subcategory == 'topological':
                    print(relation, correct)
                    show_image(pixel_values[k], captions[k])

        # print(y)
        correct_this_batch = int(torch.sum(sum_list))
        # print(correct_this_batch)
        correct += correct_this_batch
        total += len(sum_list)

    ave_score = correct / float(total)
    val_loss = tot_val_loss / len(data_loader)

    ave_per_category = {key: correct_per_category[key] / (total_per_category[key] + 1e-9) for key in unique_subcategories}
    print("Validation loss: ", val_loss)
    print("Validation acc per category: ", ave_per_category)

    ave_all = sum(correct_per_category.values()) / sum(total_per_category.values())
    return ave_score


def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the text embeddings
    for param in model.text_model.parameters():
        param.requires_grad = requires_grad


def train(args, train_loader, val_loader, model, scaler=None, step_global=0, epoch=-1, \
          val_best_score=0, processor=None, flips=[]):
    model_type = args.model_type
    train_loss = 0
    train_steps = 0

    model.cuda()
    model.train()
    acc = evaluate(val_loader, model, flips=flips, filter_relations=negate_horizontal_flip.keys())
    print("Initial validation accuracy: ", acc)


    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        if model_type == "visualbert":
            batch_cap, batch_img, y = data
            batch_inputs = {}
            for k, v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            img_attention_mask = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            img_token_type_ids = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            batch_inputs.update({
                "visual_embeds": batch_img.cuda(),
                "visual_token_type_ids": img_token_type_ids.cuda(),
                "visual_attention_mask": img_attention_mask.cuda(),
            })
        elif model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k, v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
            })
        elif model_type == "vilt":
            input_ids, pixel_values, y = data
        elif model_type == "clip":
            input_ids, pixel_values, y, captions, filenames, relations = data
        y = y.cuda()

        if args.amp:
            with autocast():
                if model_type in ["visualbert", "lxmert"]:
                    outputs = model(**batch_inputs, labels=y)
                elif model_type == "vilt":
                    outputs = model(input_ids=input_ids.cuda(),
                                    pixel_values=pixel_values.cuda(), labels=y)
                elif model_type == "clip":
                    input_ids = input_ids.view(-1, input_ids.shape[-1])
                    outputs = model(input_ids=input_ids.cuda(),
                                    pixel_values=pixel_values.cuda())

                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds

                    scores = image_features @ text_features.T # num_image * num_text

                    aligned_y = []
                    for idx, single_y in enumerate(y):
                        aligned_y.append(idx // 2 + single_y)
                    aligned_y = torch.tensor(aligned_y).cuda()
                    print(y, aligned_y)
                    text_features = text_features.view(-1, 2, text_features.shape[-1])

                    # Compute raw scores using matrix multiplication
                    scores = 100 * torch.bmm(image_features.unsqueeze(1), text_features.transpose(1, 2)).view(-1, 2)
                    scores = scores.softmax(dim=-1)
                    loss = torch.nn.CrossEntropyLoss()(scores, y.squeeze().long())
        else:
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
                loss = outputs.loss
                scores = outputs.logits
            elif model_type == "vilt":
                outputs = model(input_ids=input_ids.cuda(),
                                pixel_values=pixel_values.cuda(), labels=y)
                loss = outputs.loss
                scores = outputs.logits
            elif model_type == "clip":
                # print(input_ids.shape, pixel_values.shape)
                input_ids = input_ids.view(-1, input_ids.shape[-1])
                outputs = model(input_ids=input_ids.cuda(),
                                pixel_values=pixel_values.cuda())

                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                scores = 100 * image_features @ text_features.T  # num_image * num_text
                labels = torch.zeros_like(scores).long()

                text_features = text_features.view(-1, 2, text_features.shape[-1])
                total_loss = 0
                y = y.squeeze().tolist()
                for idx in range(len(flips) + 1):
                    selected_images = image_features[idx::len(flips) + 1]
                    selected_text = text_features[idx::len(flips) + 1].reshape(-1, text_features.shape[-1])

                    selected_labels_text = []
                    for j in range(len(selected_images)):
                        select_idx = idx + j * (len(flips) + 1)
                        selected_labels_text.append(y[select_idx] + (j * 2))

                    selected_labels_image = []
                    for j in range(len(selected_images)):
                        selected_labels_image.append(j)

                    selected_labels_text = torch.tensor(selected_labels_text).cuda().long()
                    selected_labels_image = torch.tensor(selected_labels_image).cuda().long()

                    text_scores = 100 * selected_images @ selected_text.T
                    image_scores = 100 * selected_text[selected_labels_text] @ selected_images.T

                    text_loss = F.cross_entropy(text_scores.softmax(-1), selected_labels_text)
                    image_loss = F.cross_entropy(image_scores.softmax(-1), selected_labels_image)
                    total_loss += text_loss + image_loss

                loss = total_loss

        if args.amp:
            scaler.scale(loss).backward()
            if train_steps % args.cumulative_grad_steps == 0:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if train_steps % args.cumulative_grad_steps == 0:
                optimizer.step()

        # log lr
        lr = optimizer.param_groups[0]['lr']

        train_loss += loss.item()
        train_steps += 1
        step_global += 1

        # evaluate and save
        if step_global % args.eval_step == 0:
            # evaluate
            acc = evaluate(val_loader, model, flips=flips, filter_relations=negate_horizontal_flip.keys())
            print(f"====== evaliuate ======")
            print(f"epoch: {epoch}, global step: {step_global}, val performance: {acc}")
            print(f"=======================")
            if val_best_score < acc:
                val_best_score = acc
            else:
                continue
            checkpoint_dir = os.path.join(args.output_dir, f"best_checkpoint")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if model_type == "visualbert":
                model.save_pretrained(checkpoint_dir)
            elif model_type == "lxmert":
                model.lxmert.save_pretrained(checkpoint_dir)
            elif model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
            elif model_type == "clip":
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ckpt.pt"))

            print(f"===== best model saved! =======")

    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global, val_best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--img_feature_path', type=str, default='../data/trainval2017')
    parser.add_argument('--train_json_path', type=str, default="/home/server17/taewon_workspace/ai502/data/splits/random/train.jsonl")
    parser.add_argument('--val_json_path', type=str, default="/home/server17/taewon_workspace/ai502/data/splits/random/test.jsonl")
    parser.add_argument('--model_type', type=str, default="clip", help="visualbert or lxmert or vilt")
    parser.add_argument('--model_path', type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--amp', action="store_true", \
                        help="automatic mixed precision training")
    parser.add_argument('--output_dir', type=str, default="clip_full_tune")
    parser.add_argument('--checkpoint_step', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--wandb', action="store_true", help="use wandb")
    parser.add_argument('--data_parallel', action="store_true", help="use data parallel")
    parser.add_argument('--cumulative_grad_steps', type=int, default=1)
    parser.add_argument('--flip_handler', type=str, default='none')
    parser.add_argument('--flips', nargs='+', default=[], help='A list of texts')

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    model_type = args.model_type
    # load model
    if model_type == "visualb ert":
        model = VisualBertForVisualReasoning.from_pretrained(args.model_path)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        processor = None
    elif model_type == "lxmert":
        model = LxmertForPreTraining.from_pretrained(args.model_path)
        model = LxmertForBinaryClassification(model)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        processor = None
    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltModel, ViltForImagesAndTextClassification

        config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm")
        config.num_images = 1
        model = ViltForImagesAndTextClassification(config)
        model.vilt = ViltModel.from_pretrained(args.model_path)
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        tokenizer = None
    elif model_type == "clip":
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = AutoModel.from_pretrained(args.model_path)
        set_grad(model, requires_grad=True)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # load data
    def collate_fn_batch_visualbert(batch):
        captions, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions),
            max_length=32,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        labels = torch.tensor(labels)
        return toks, img_features, labels


    def collate_fn_batch_lxmert(batch):
        captions, boxes, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions),
            max_length=32,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        boxes = torch.stack(boxes)
        labels = torch.tensor(labels)
        return toks, boxes, img_features, labels


    def collate_fn_batch_vilt(batch):
        # """
        imgs, captions, labels = zip(*batch)
        inputs = processor(images=list(imgs), text=list(captions), return_tensors="pt",
                           padding='max_length', truncation=True, add_special_tokens=True)
        # """
        # print (inputs.input_ids.shape, inputs.pixel_values.shape)
        """
        inputs, labels = zip(*batch)
        inputs_ids = [i.input_ids for i in inputs]
        pixel_values = [i.pixel_values for i in inputs]
        for i in pixel_values:
            print (i.shape)
        """
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values, labels
        # return torch.cat(inputs_ids, dim=0), torch.cat(pixel_values, dim=0), labels


    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames, relations = zip(*batch)
        # print(labels)
        input_id_list = []
        pixel_value_list = []
        labels_list = []
        caption_list = []
        for item_idx in range(len(imgs)):
            inputs = processor(captions[item_idx], images=imgs[item_idx], return_tensors="pt", padding='max_length')
            input_ids = inputs.input_ids
            pixel_values = inputs.pixel_values


            # pixel_values = pixel_values.repeat(input_ids.shape[0] // (2 * pixel_values.shape[0]), 1, 1, 1)

            # Create a tensor from labels[item_idx] and cast it to torch.int8
            expanded_label = torch.tensor(labels[item_idx], dtype=torch.int8).repeat(input_ids.shape[0] // 2, 1)
            # print(expanded_label)

            input_id_list.append(input_ids)
            pixel_value_list.append(pixel_values)
            labels_list.append(expanded_label)
            caption_list.append(captions[item_idx])
        input_ids = torch.cat(input_id_list, dim=0).view(-1, 2, input_ids.shape[-1])
        pixel_values = torch.cat(pixel_value_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return input_ids, pixel_values, labels, caption_list, filenames, relations


    img_feature_path = args.img_feature_path
    # relations = list(negate_horizontal_flip.keys())
    relations = None
    visual_json_path = os.path.join('../data', 'annotations', 'panoptic_train2017.json')
    dataset_train = ImageTextClassificationDataset(
        img_feature_path,
        args.train_json_path,
        filter_relations=relations,
        flips=args.flips,
        remaining_flip_handler=args.flip_handler,
        visual_prompt=False,
        visual_json_path=visual_json_path
    )
    dataset_val = ImageTextClassificationDataset(
        img_feature_path,
        args.val_json_path,
        filter_relations=relations,
        flips=args.flips,
        remaining_flip_handler=args.flip_handler,
        visual_prompt=False,
        visual_json_path=visual_json_path
    )

    if model_type == "visualbert":
        collate_fn_batch = collate_fn_batch_visualbert
    elif model_type == "lxmert":
        collate_fn_batch = collate_fn_batch_lxmert
    elif model_type == "vilt":
        collate_fn_batch = collate_fn_batch_vilt
    elif model_type == "clip":
        collate_fn_batch = collate_fn_batch_clip

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn=collate_fn_batch,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        collate_fn=collate_fn_batch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=True
    )

    # mixed precision training
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    global_step, val_best_score = 0, 0
    for epoch in range(args.epoch):
        loss, global_step, val_best_score = train(args, train_loader, val_loader, model, scaler=scaler, \
                                                  step_global=global_step, epoch=epoch, val_best_score=val_best_score,
                                                  processor=processor, flips=args.flips)
        print(f"epoch: {epoch}, global step: {global_step}, loss: {loss}")
