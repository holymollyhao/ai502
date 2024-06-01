
import os
import argparse
from tqdm.auto import tqdm
import torch
from data import ImageTextClassificationDataset, negate_horizontal_flip

def evaluate(data_loader, model, is_original=False):
    model.cuda()
    model.eval()

    correct_orig = 0
    correct, total, all_true = 0, 0, 0
    preds = []
    errors = list()
    inital_vs_final_disagreements = 0
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        input_ids, pixel_values, y, captions, filenames = data
        y = y.cuda()
        with torch.no_grad():
            batch_cap = input_ids.cuda()
            batch_img = pixel_values.cuda()
            outputs = model(input_ids=batch_cap, 
                pixel_values=batch_img)

        # reproduce huggingface webapp
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        preds_vote = []
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

        orig_preds = torch.argmax(image_features @ text_features[:2].T, dim=1)[0]
        orig_max_prob = torch.max((image_features @ text_features[:2].T)[0])
        aug_preds = torch.argmax(image_features @ text_features[2:].T, dim=1)[1]
        aug_max_prob = torch.max((image_features @ text_features[2:].T)[1])

        if orig_max_prob < aug_max_prob:
            preds_current = aug_preds
        else:
            preds_current = orig_preds


        if orig_preds != aug_preds:
            if orig_preds == y:
                print("orig correct")
            elif aug_preds == y:
                print("aug correct")
            print(filenames[0], captions[0], captions[1], orig_preds.item(), aug_preds.item(), y.item(), preds_current.item())

        correct_this_batch = int(sum(y == preds_current))
        correct_this_batch_orig = int(sum(y == orig_preds))
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
        total+=batch_img.shape[0]
        all_true += sum(y)
    ave_score = correct / float(total)
    ave_score_orig = correct_orig / float(total)
        # if correct_this_batch != batch_img.shape[0]:
        #     errors.append(filenames[0]+' '+str(int(y[0]))+' '+captions[0]+', '+captions[1]+', '+str(float(scores[0][0]))+', '+str(float(scores[0][1])))


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

    model_url = args.model_url
    # load model
    print("Loading CLIP model...")
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(model_url)
    model = AutoModel.from_pretrained(model_url)
    
    json_path = os.path.join('data', 'data_files', 'all_vsr_validated_data.jsonl') # 10119 image text pairs
    img_path = os.path.join('data', 'images') # changes to images
    relations = list(negate_horizontal_flip.keys())
    dataset = ImageTextClassificationDataset(
        os.path.join('../', img_path),
        os.path.join('../', json_path),
        filter_relations=relations,
        flips=['adaptive_flip']
    )
    # dataset = ImageTextClassificationDataset(img_path, json_path, filter_relations=relations, flips=[])


    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames = zip(*batch)

        inputs = processor(captions[0], images=imgs[0], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids
        pixel_values = inputs.pixel_values
        # print(input_ids.shape, pixel_values.shape)
        
        labels = torch.tensor(labels)
        return input_ids, pixel_values, labels, list(captions[0]), filenames
        
    collate_fn_batch = collate_fn_batch_clip

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=1,
        shuffle=False,
        num_workers=0,)
    orig_acc, acc, total, all_true, preds, errors = evaluate(test_loader, model, args.is_original)
    print (f"total example: {total}, # true example: {all_true}, orig_acc: {orig_acc}, acccuracy: {acc}")

    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")

