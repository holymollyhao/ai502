
import os
import argparse
from tqdm.auto import tqdm
import torch
from data import ImageTextClassificationDataset

def evaluate(data_loader, model):
    model.cuda()
    model.eval()

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

        raw_scores = (100.0 * image_features @ text_features.T)

        scores = torch.as_tensor([[raw_scores[0][0], raw_scores[0][1]]]).cuda()
        initial_preds_current = torch.argmax(scores, dim=1)
        initial_correct_this_batch = int(sum(y == initial_preds_current))

        scores = raw_scores
        scores = scores.softmax(dim=-1)
        
        preds_current = torch.argmax(scores, dim=1) # choose the higher similarity one
        correct_this_batch = int(sum(y == preds_current))

        if initial_correct_this_batch != correct_this_batch:
            inital_vs_final_disagreements += 1
        correct += correct_this_batch
        preds += preds_current.cpu().numpy().tolist()
        total+=batch_img.shape[0]
        all_true += sum(y)
        ave_score = correct / float(total)
        if correct_this_batch != batch_img.shape[0]:
            errors.append(filenames[0]+' '+str(int(y[0]))+' '+captions[0]+', '+captions[1]+', '+str(float(scores[0][0]))+', '+str(float(scores[0][1])))


    # TODO: save also predictions
    return ave_score, total, all_true, preds, errors
            

if __name__ == "__main__":
    # takes 10 min for evaluation
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--model_url', type=str, default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument('--output_preds', action='store_true')

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
    dataset = ImageTextClassificationDataset(img_path, json_path)

    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames = zip(*batch)
        inputs = processor(captions[0], images=imgs, return_tensors="pt", padding=True)
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values, labels, list(captions[0]), filenames
        
    collate_fn_batch = collate_fn_batch_clip

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=1,
        shuffle=False,
        num_workers=0,)
    acc, total, all_true, preds, errors = evaluate(test_loader, model)
    print (f"total example: {total}, # true example: {all_true}, acccuracy: {acc}")

    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")

