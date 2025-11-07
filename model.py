import torch
import torch.nn as nn
import math

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, input_ids, token_type_ids=None):
        bsz, seq_len = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(position_ids)
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = words + positions + token_types
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size%num_heads == 0, "hidden size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(hidden_size, 3*hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask = None):
        bsz, seq_len, _ = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def shape_for_heads(tensor):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        q = shape_for_heads(q)
        k = shape_for_heads(k)
        v = shape_for_heads(v)

        scores = torch.matmul(q, k.transpose(-2,-1))/self.scale

        if attention_mask is not None:
            if attention_mask.dim() ==2:
                attn = attention_mask[:, None, None, :].to(dtype = scores.dtype)
                attn = (1.0-attn)*-1e9
                scores = scores+attn
            else:
                scores = scores+attention_mask

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.out_proj(context)
        out = self.proj_dropout(out)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.act(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attn_out = self.attn(x, attention_mask=attention_mask)
        x = self.attn_layer_norm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.ffn_layer_norm(x+self.dropout(ffn_out))
        return x
        

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)]
        )
    
    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x

class SimpleBertForPreTraining(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size, max_position_embeddings, type_vocab_size, dropout):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout)
        self.encoder = TransformerEncoder(num_layers, hidden_size, num_heads, intermediate_size, dropout)

        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler_act = nn.Tanh()

        self.mlm_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=1e-12)
        )

        self.mlm_decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        self.nsp_classifier = nn.Linear(hidden_size, 2)

        self.apply(init_weights)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_mask is None:
            attention_mask = (input_ids!=0).long()
        
        extended_attention_mask = attention_mask[:, None, None, :].to(dtype = embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask)* -1e9

        sequence_output = self.encoder(embedding_output, attention_mask = extended_attention_mask)

        pooled_output = self.pooler(sequence_output[:,0])
        pooled_output = self.pooler_act(pooled_output)

        mlm_hidden = self.mlm_transform(sequence_output)

        prediction_scores = torch.matmul(mlm_hidden, self.embeddings.word_embeddings.weight.t()) + self.mlm_decoder_bias

        seq_relationship_score  = self.nsp_classifier(pooled_output)

        return prediction_scores, seq_relationship_score



def _test_small_forward():
    vocab_size = 30522
    batch_size = 2
    seq_len = 32

    model = SimpleBertForPreTraining(vocab_size=vocab_size,
                                     hidden_size=256,
                                     num_layers=2,
                                     num_heads=8,
                                     intermediate_size=512,
                                     max_position_embeddings=512,
                                     type_vocab_size  = 2,
                                     dropout=0.1)
    
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    pred_scores, nsp_scores = model(input_ids, token_type_ids, attention_mask)

    
    assert pred_scores.shape == (batch_size, seq_len, vocab_size), f"mlm shape mismatch: {pred_scores.shape}"
    assert nsp_scores.shape == (batch_size, 2), f"nsp shape mismatch: {nsp_scores.shape}"

    print("prediction_scores shape:", pred_scores.shape)
    print("nsp_scores shape:", nsp_scores.shape)
    print("Test passed")
    
    

if __name__ == "__main__":
    _test_small_forward()