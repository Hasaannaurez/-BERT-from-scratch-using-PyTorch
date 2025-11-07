from transformers import BertTokenizerFast
from datasets import load_dataset
import nltk
from tqdm import tqdm
import random
from torch.utils.data import Dataset
import torch

def load_and_split_wikitext_2():
    print("inside load_and_split_wikitext_2")
    ds = load_dataset("wikitext", "wikitext-2-v1", split = "train")
    print("Successfully loaded wiki in load_and_split_wikitext_2")

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    documents = []
    for text in tqdm(ds["text"], desc="Processing"):
        test = text.strip()

        if not text:
            # print("No text found in load_and_split_wikitext_2")
            continue

        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            documents.append(sentences)

    print(f"Loaded {len(documents)} documents")
    return documents

def build_sentence_pairs(documents, negative_ratio):
    all_pairs = []
    num_docs = len(documents)

    docs_with_pairs = []
    doc_indices = []
    for doc_idx, sents in enumerate(documents):
        if(len(sents)>=2):
            docs_with_pairs.append(doc_idx)

        for s_idx in range(len(sents)):
            doc_indices.append((doc_idx, s_idx))

    for doc_idx in tqdm(docs_with_pairs, desc="Docs w >=2 sents"):
        sents = documents[doc_idx]
        for i in range(len(sents)-1):
            a = sents[i]
            b_pos = sents[i+1]
            all_pairs.append((a, b_pos, 1))
        
            for _ in range(negative_ratio):
                rand_doc_idx = doc_idx
                while rand_doc_idx == doc_idx:
                    rand_doc_idx = random.randrange(num_docs)
                
                b_neg = random.choice(documents[rand_doc_idx])
                all_pairs.append((a, b_neg, 0))

    return all_pairs

def create_masked_in_labels(input_ids, tokenizer, mlm_probablity):
    labels = input_ids.clone()
    special_mask = torch.tensor(tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True), dtype=torch.bool)

    probablity_matrix = torch.full(labels.shape, mlm_probablity)
    probablity_matrix[special_mask] = 0.0

    masked_indices = torch.bernoulli(probablity_matrix).bool()

    if masked_indices.sum().item() == 0:
        candidate_positions = (~special_mask).nonzero(as_tuple=False).view(-1).tolist()
        if len(candidate_positions) > 0:
            rand_pos = random.choice(candidate_positions)
            masked_indices[rand_pos] = True

        
    labels[~masked_indices] = -100

    masked_positions = masked_indices.nonzero(as_tuple=False).squeeze(-1).tolist()

    input_ids_masked = input_ids.clone()

    if(len(masked_positions)==0):
        print("Lenght of masked_positions = 0, in create_masked_in_labels")
        return input_ids_masked, labels
    
    for pos in masked_positions:
        prob = random.random()
        if prob < 0.8:
            input_ids_masked[pos] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        elif prob < 0.9:
            rand_id = random.randint(0, tokenizer.vocab_size-1)
            input_ids_masked[pos] = rand_id
        else:
            pass
            
    return input_ids_masked, labels

class BertPretrainingDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_seq_length):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        tuple_pair = self.pairs[idx]
        a = tuple_pair[0]
        b = tuple_pair[1]
        is_next = tuple_pair[2]

        enc = self.tokenizer(a, b, 
                             add_special_tokens = True, 
                             max_length = self.max_seq_length, 
                             padding = 'max_length',
                             truncation = True,
                             return_attention_mask = True,
                             return_token_type_ids = True)
        
        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        token_type_ids = torch.tensor(enc['token_type_ids'], dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long) 
        nsp_label = torch.tensor(is_next, dtype=torch.long)


        masked_input_ids, mlm_labels = create_masked_in_labels(input_ids, self.tokenizer, mlm_probablity = 0.15)

        return {
            'input_ids': masked_input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            "nsp_label": nsp_label,
            "mlm_labels": mlm_labels
        }


def small_test():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    docs = load_and_split_wikitext_2()

    pairs = build_sentence_pairs(docs, negative_ratio=1)
    
    ds = BertPretrainingDataset(pairs[:200], tokenizer, max_seq_length=128)

    print("Printing sample tokenized+masked examples")

    for i in range(3):
        sample = ds[i]
        print(f"Example {i}: NSP label = {sample['nsp_label'].item()}")
        print("input_ids (masked) : ", sample['input_ids'].tolist())
        print("tokens (masked) : ", tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist()))

        mlm_labels = sample['mlm_labels']
        masked_positions = (mlm_labels != -100).nonzero(as_tuple=False).squeeze(-1).tolist()

        print("masked_positions : ", masked_positions)

        if isinstance(masked_positions, int):
            masked_positions = [masked_positions]
        
        for p in masked_positions:
            orig_id = mlm_labels[p].item()
            print(f"pos {p} : label id = {orig_id}, token={tokenizer.convert_ids_to_tokens(orig_id)}")

        print("token_type_ids :", sample['token_type_ids'].tolist())
        print("attention_mask : ", sample['attention_mask'].tolist())
        print("-"*30)



if __name__ == "__main__":
    small_test()