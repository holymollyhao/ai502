
import os
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from data import ImageTextClassificationDataset, negate_horizontal_flip
from copy import deepcopy

def show_image(image, caption):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
    plt.title(caption)
    plt.show()


def draw_tsne(features):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # Convert features to numpy array if they are not already
    features = features.cpu().detach().numpy()

    # Generate labels alternating between 0 and 1
    labels = np.array([i % 2 for i in range(len(features))])

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # Create scatter plot, coloring by labels
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.show()


def set_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the text embeddings
    for param in model.text_model.embeddings.token_embedding.parameters():
        param.requires_grad = requires_grad

    for param in model.text_model.embeddings.position_embedding.parameters():
        param.requires_grad = requires_grad

def take_step(model, input_ids, pixel_values):
    model.zero_grad()
    model.train()
    set_grad(model, True)
    input_ids, pixel_values = input_ids.cuda(), pixel_values.cuda()
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds.view(-1, 2, outputs.text_embeds.shape[1])
    pred_entropy = 0
    for idx, text_feat in enumerate(text_features):
        if idx + 1 <= len(text_features) / 2:
            im_feat = image_features[0]
        else:
            im_feat = image_features[1]
        raw_scores = (100.0 * im_feat @ text_feat.T)
        scores = raw_scores.softmax(dim=-1)
        pred = torch.argmax(scores, dim=-1)
        pred_entropy += -torch.sum(scores * torch.log(scores + 1e-9), dim=-1)

    loss = pred_entropy
    loss.backward()
    optimizer.step()
    return model

def evaluate(data_loader, model, flips=None):
    model.cuda()
    model.eval()

    correct_orig = 0
    correct, total, all_true = 0, 0, 0
    preds = []
    errors = list()
    inital_vs_final_disagreements = 0

    # initial_state_dict = deepcopy(model.state_dict())

    image_feat_list = []
    text_feat_list = []

    correct_per_relation = {}
    cnt_per_relation = {}
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids, pixel_values, y, captions, filenames, relation_names = data
        y = y.cuda()

        # model.load_state_dict(initial_state_dict)
        # model = take_step(model, input_ids, pixel_values)

        with torch.no_grad():
            batch_cap = input_ids.cuda()
            batch_img = pixel_values.cuda()
            outputs = model(input_ids=batch_cap, 
                pixel_values=batch_img)

        # reproduce huggingface webapp
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        preds_vote = []

        # if relation_names[0] in ['above', 'below']:
        if relation_names[0]  in ['at the right side of', 'at the left side of', 'left of', 'right of']:
            image_feat_list.append(image_features)

            text_feat_list.append(text_features[:len(text_features)//2])

        if 'synonym' in flips:
            score_list = []
            pred_list = []
            entropy_list = []

            reshaped_text_features = text_features.view(-1, 2, text_features.shape[1])

            for idx, text_feat in enumerate(reshaped_text_features):
                if idx + 1 <= len(reshaped_text_features)/2:
                    im_feat = image_features[0]
                else:
                    im_feat = image_features[1]
                raw_scores = (100.0 * im_feat @ text_feat.T)
                scores = raw_scores.softmax(dim=-1)
                pred = torch.argmax(scores, dim=-1)
                pred_entropy = -torch.sum(scores * torch.log(scores + 1e-9), dim=-1)

                score_list.append(scores)
                pred_list.append(pred)
                entropy_list.append(pred_entropy)

            score_list = torch.stack(score_list)
            pred_list = torch.stack(pred_list)


            entropy_list = torch.reciprocal(torch.stack(entropy_list) + 1e-9)
            prob_entropy = F.softmax(entropy_list * 10, dim=0)

            lowest_entropy_idx = torch.argmax(entropy_list)
            weighted_mean_avg = torch.sum(pred_list.T * prob_entropy, dim=0)
            weighted_mean_avg_int = torch.round(weighted_mean_avg).to(torch.int)

            print(pred_list)
            print(entropy_list)
            preds_current = pred_list[lowest_entropy_idx]

            # if int(sum(y != preds_current)):
            #     show_image(pixel_values[0], captions[0])
        else:
            for voting in range(image_features.shape[0]):
                image_feature = image_features[voting, :].unsqueeze(dim=0)
                text_feature = text_features[2*voting:2*voting+2, :]
                # print(image_feature.shape, text_feature.shape)
                raw_scores = (100.0 * image_feature @ text_feature.T)
                # print(raw_scores)
                scores = torch.as_tensor([[raw_scores[0][0], raw_scores[0][1]]]).cuda()
                scores = scores.softmax(dim=-1)
                preds_current = torch.argmax(scores, dim=1)
                preds_vote.append(preds_current)
            preds_vote = torch.Tensor(preds_vote)
            preds_current = torch.Tensor([torch.median(preds_vote)]).cuda()

        orig_scores = (100.0 * image_features[0] @ text_features[:2].T)
        orig_scores = orig_scores.softmax(dim=-1)
        orig_preds = torch.argmax(orig_scores, dim=0)

        correct_this_batch_orig = int(sum(y == orig_preds))


        # aug_scores = (100.0 * image_features[1] @ text_features[2:].T)
        # aug_scores = aug_scores.softmax(dim=-1)
        # aug_preds = torch.argmax(aug_scores, dim=-1)

        # if max(aug_scores) > max(orig_scores):
        #     preds_current = torch.Tensor([1]).squeeze()
        # else:
        #     preds_current = torch.Tensor([0]).squeeze()

        # if orig_preds != aug_preds:
        #     preds_current = torch.Tensor([0]).squeeze()
        # else:
        #     preds_current = torch.Tensor([1]).squeeze()

        correct_this_batch = int(sum(y == preds_current))

        if relation_names[0] in correct_per_relation.keys():
            correct_per_relation[relation_names[0]] += correct_this_batch
            cnt_per_relation[relation_names[0]] += 1
        else:
            correct_per_relation[relation_names[0]] = correct_this_batch
            cnt_per_relation[relation_names[0]] = 1

        # if int(sum(y != preds_current)):
        #     show_image(pixel_values[0], captions[0])

        # print(preds_current, y, correct_this_batch)
        # print(preds_vote, preds_current, y)
        # print(y, correct_this_batch)
        # exit()
            
        # print(image_features.shape, text_features.shape)
        # import pdb; pdb.set_trace()
        # raw_scores = (100.0 * image_features @ text_features.T)

        # scores = torch.as_tensor([[raw_scores[0][0], raw_scores[0][1]]]).cuda()
        # initial_preds_current = torch.argmax(scores, dim=1)
        # initial_correct_this_batch = int(sum(y == initial_preds_current))

        # scores = raw_scores
        # scores = scores.softmax(dim=-1)
        
        # preds_current = torch.argmax(scores, dim=1) # choose the higher similarity one
        # correct_this_batch = int(sum(y == preds_current))

        # if initial_correct_this_batch != correct_this_batch:
        #     inital_vs_final_disagreements += 1
        
        correct += correct_this_batch
        correct_orig += correct_this_batch_orig
        preds += [preds_current.cpu().numpy()]
        total+=y.shape[0]
        all_true += sum(y)
        print(f'average: {correct_orig/total}, current: {correct/total}')

        for key in correct_per_relation.keys():
            import numpy as np
            print(f'{key}: {np.round(correct_per_relation[key] / cnt_per_relation[key], 2)}')

        # if len(text_feat_list) == 100:
        #     break
        # print(f'per relation: {correct_per_relation / total}')
    ave_score = correct / float(total)
    ave_score_orig = correct_orig / float(total)
        # if correct_this_batch != batch_img.shape[0]:
        #     errors.append(filenames[0]+' '+str(int(y[0]))+' '+captions[0]+', '+captions[1]+', '+str(float(scores[0][0]))+', '+str(float(scores[0][1])))

    image_feat_list = torch.cat(image_feat_list, dim=0)
    # draw_tsne(image_feat_list)
    draw_tsne(image_feat_list)
    # TODO: save also predictions
    return ave_score_orig, ave_score, total, all_true, preds, errors
            

if __name__ == "__main__":
    # takes 10 min for evaluation
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--model_url', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--output_preds', action='store_true')
    parser.add_argument('--is_original', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visual_json_path = os.path.join('../data', 'annotations', 'panoptic_train2017.json')
    model_url = args.model_url
    # load model
    print("Loading CLIP model...")
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(model_url)
    model = AutoModel.from_pretrained(model_url)

    flips = ['adaptive_flip']
    
    json_path = os.path.join('data', 'data_files', 'all_vsr_validated_data.jsonl') # 10119 image text pairs
    img_path = os.path.join('data', 'images') # changes to images
    relations = list(negate_horizontal_flip.keys())
    dataset = ImageTextClassificationDataset(
        os.path.join('../', img_path),
        os.path.join('../', json_path),
        filter_relations=relations,
        flips=flips,
        visual_prompt=False,
        synonym_objects=True,
        visual_json_path=visual_json_path,
    )
    # dataset = ImageTextClassificationDataset(img_path, json_path, filter_relations=relations, flips=[])


    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames, relations = zip(*batch)

        inputs = processor(captions[0], images=imgs[0], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values
        # print(input_ids.shape, pixel_values.shape)

        labels = torch.tensor(labels)
        return input_ids, pixel_values, labels, list(captions[0]), filenames, relations
        
    collate_fn_batch = collate_fn_batch_clip

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=1,
        shuffle=False,
        num_workers=0,)
    orig_acc, acc, total, all_true, preds, errors = evaluate(test_loader, model, flips=flips)
    print (f"total example: {total}, # true example: {all_true}, orig_acc: {orig_acc}, acccuracy: {acc}")

    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")

