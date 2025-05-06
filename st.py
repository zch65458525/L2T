from sentence_transformers import SentenceTransformer
import torch
class SentenceEncoder:
    def __init__(self, name, root="cache_data/model", batch_size=1, multi_gpu=False):
        self.name = name
        self.root = root
        self.device = torch.device('cuda:0')
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = SentenceTransformer("multi-qa-distilbert-cos-v1", device=self.device, cache_folder=self.root, )
        self.encode = self.ST_encode
    def ST_encode(self, texts, to_tensor=True):
        if self.multi_gpu:
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True,
                convert_to_tensor=to_tensor, convert_to_numpy=not to_tensor, )
        return embeddings
    def llama_encode(self, texts, to_tensor=True):

        self.tokenizer.pad_token = self.tokenizer.eos_token
        all_embeddings = []
        with torch.no_grad(): 
            for start_index in range(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                input_ids = self.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).input_ids.to(self.device)
                transformer_output = self.model(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"]
                word_embeddings = transformer_output[-1].detach()
                sentence_embeddings = word_embeddings.mean(dim=1)
                all_embeddings.append(sentence_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings