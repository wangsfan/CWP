import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

import copy
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torch.nn.functional as F

def load_clip_to_cpu():
    url = clip._MODELS['ViT-L/14']
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict()).eval()
    return model.float()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, args = None,clip_model =None,class_list=None):
        super().__init__()

        #prompt_learner-----------------------
        dtype = clip_model.dtype


        self.classes_list = class_list
        self.class_list = [self.classes_list[i:i + 20] for i in range(0, 200, 20)]
        self.class_task_1 = [' '.join(itm) for itm in self.class_list]
        self.class_task = ['a photo of ' + itm for itm in self.class_task_1]

        self.class_task_class = ['a photo of ' + a for a in self.classes_list]
        self.e_pool_size = 200
        self.n_tasks = 10
        self.task_count = 0
        self.old_num_k = 0
        self.new_num_k = 0

        # self.token_embedding = token_embedding
        n_cls = len(self.class_task)
        n_cls_class = len(self.class_task_class)

        n_ctx = 20
        ctx_init = False
        ctx_dim = 768

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if True:
                print("Initializing class-specific contexts")
                ctx_class_vectors = torch.empty(n_cls_class, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_class_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")


        self.ct_class = torch.nn.Parameter(ctx_class_vectors, requires_grad=True)

        #class_level prompt_learner-----------------------
        classnames_class = [name.replace("_", " ") for name in self.class_task_class]
        prompt_class = [prompt_prefix + " " + name + "." for name in classnames_class]
        tokenized_prompts_class = torch.cat([clip.tokenize(p) for p in prompt_class])
        with torch.no_grad():
            embedding_class = clip_model.token_embedding(tokenized_prompts_class).type(dtype)

        self.register_buffer("token_prefix_class", embedding_class[:, :1])  # SOS
        self.register_buffer("token_suffix_class", embedding_class[:, 1 + n_ctx :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.tokenized_prompts_class = tokenized_prompts_class  # torch.Tensor

        self.class_token_position = 'end'
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def process_task_count(self, old_num_k, new_num_k):
        self.task_count += 1
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k

        self.ct_class = self.gram_schmidt(self.ct_class)

    def forward(self):

        ctx_class = self.ct_class

        prefix_class = self.token_prefix_class
        suffix_class = self.token_suffix_class

        if self.class_token_position == "end":

            prompts_class = torch.cat(
                [
                    prefix_class,  # (n_cls, 1, dim)
                    ctx_class,     # (n_cls, n_ctx, dim)
                    suffix_class,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError

        return prompts_class
    


def get_out_batch(batch_size, task_mean, task_std):
    out = []
    for i in range(batch_size):
        out.append(task_mean + task_std * torch.randn_like(task_mean))
    return torch.stack(out).to(task_mean.device)



class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, num_tasks=10, kernel_size=17, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 prompts_per_task=5,class_list=None,args=None):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.n_tasks = num_tasks
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.use_prompt_embed_matcher = False
        self.prompts_per_task = prompts_per_task
        self.old_num_k = 0
        self.new_num_k = 0
        self.sigmoid = nn.Sigmoid()

        self.ker_size = kernel_size
        self.stride = 1
        self.dilation = 1
        self.conv_channels = 1 


        # This is the max number of kernels (self.pool_size) that can occur if we use max number of prompts per task
        self.pool_size = self.n_tasks * prompts_per_task
        print("Num Tasks: ", self.n_tasks, "pool_size: ", self.pool_size, "kernel_size: ", kernel_size, "top_k: ", top_k)
        if self.use_prompt_embed_matcher:
            self.prompt_embed_matcher = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(embed_dim, embed_dim // 2)),
                ('relu1', nn.ReLU()),
                ('linear2', nn.Linear(embed_dim // 2, embed_dim // 4))
            ]))

        if self.use_prefix_tune_for_e_prompt:
            assert embed_dim % self.num_heads == 0
            if self.same_key_value:
                prompt_pool_shape = (self.num_layers, self.length + self.ker_size - 1, 
                                    embed_dim + (self.ker_size - 1) * self.num_heads)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
            
            else:
                prompt_pool_shape = (self.num_layers, (self.length + self.ker_size - 1) * 2, 
                                    embed_dim + (self.ker_size - 1) * self.num_heads)
                prompt_token_pool_shape = (self.pool_size, 2*self.length, embed_dim)

                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':

                    for i in range(self.num_layers):
                        prompt_token = nn.Parameter(torch.zeros(prompt_token_pool_shape))
                        nn.init.uniform_(prompt_token, -1, 1)
                        setattr(self, f'e_p_{i}',prompt_token)

                elif prompt_init == 'ortho':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    for i in range(self.num_layers):
                        nn.init.orthogonal_(self.prompt[i])

        else:
            prompt_pool_shape=(self.num_layers, self.length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            elif prompt_init == 'ortho':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.orthogonal_(self.prompt)
                    
        # if using learnable prompt keys
        if prompt_key:
            if self.use_prompt_embed_matcher:
                key_shape = (self.pool_size, embed_dim // 4)
            else:
                key_shape = (self.pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
            elif prompt_key_init == 'ortho':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.orthogonal_(self.prompt_key)
                
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def process_new_task(self, old_num_k, new_num_k):
        self.old_num_k = old_num_k
        self.new_num_k = new_num_k
        print("Old Num K: ", self.old_num_k, "New Num K: ", self.new_num_k)
        self.prompt_key = self.gram_schmidt(self.prompt_key)

        for e in range(self.num_layers):
            p_1 = getattr(self,f'e_p_{e}') # 0 based indexing here
            p_2 = self.gram_schmidt(p_1)
            setattr(self, f'e_p_{e}',p_2)


    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point

        s = self.old_num_k
        f = self.new_num_k
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, task_id=-1, prompt_mask=None, layer_num= -1, cls_features=None,
                text_features_class=None,text_features_task=None,cls_features_task=None):
        out = dict()

        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
            

            if self.use_prompt_embed_matcher:
                x_embed_mean = self.prompt_embed_matcher(x_embed_mean)

            if self.use_prompt_embed_matcher:
                text_features_class = self.prompt_embed_matcher(text_features_class)
                text_features_task = self.prompt_embed_matcher(text_features_task)

            s = self.old_num_k
            f = self.new_num_k
            prompt_token_1 = getattr(self,f'e_p_{layer_num}')

            if self.training:

                prompt_key = self.prompt_key[task_id*self.prompts_per_task:(task_id+1)*self.prompts_per_task]
                prompt_token = prompt_token_1[task_id*self.prompts_per_task:(task_id+1)*self.prompts_per_task]
                text_features_class = text_features_class[task_id*self.prompts_per_task:(task_id+1)*self.prompts_per_task]
                text_features_task_ = text_features_task[task_id]

                prompt_key_norm = self.l2_normalize(text_features_task, dim=-1) # Pool_size, C
                x_embed_norm = self.l2_normalize(cls_features_task, dim=-1) # B, C
            
                similarity_1 = torch.matmul(prompt_key_norm[task_id], x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
                similarity_2 = similarity_1.t() # B, pool_size
                out['similarity_task'] = similarity_2
 
            else:

                prompt_key_norm = self.l2_normalize(text_features_task, dim=-1) # Pool_size, C
                x_embed_norm = self.l2_normalize(cls_features_task, dim=-1) # B, C
            
                similarity_1 = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
                similarity_2 = similarity_1.t() # B, pool_size
                (similarity_top_k, idx) = torch.topk(similarity_2, k=self.top_k, dim=1) # B, top_k

                if True:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)

                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
          
                
                out['prompt_idx'] = idx
                prompt_key = self.prompt_key[idx[0]*self.prompts_per_task:(idx[0]+1)*self.prompts_per_task]
                prompt_token = prompt_token_1[idx[0]*self.prompts_per_task:(idx[0]+1)*self.prompts_per_task]
                text_features_class = text_features_class[idx[0]*self.prompts_per_task:(idx[0]+1)*self.prompts_per_task]

            prompt_text_features_class_norm = self.l2_normalize(text_features_class, dim=-1)
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
        
            similarity = torch.matmul(prompt_text_features_class_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size


            # similarity
            out['similarity'] = similarity


            if self.use_prefix_tune_for_e_prompt:
                prompt_token = prompt_token.unsqueeze(0).repeat(x_embed.shape[0], 1, 1, 1)

            else:
                batched_prompt_raw = self.prompt[layer_num,:]
                batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(
                    batch_size, top_k * length, embed_dim
                )

            out['x_embed_norm'] = x_embed_norm

        else:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0

                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)

                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length, 
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)

            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        batched_prompt = self.compute_multi_prompt(prompt_token, f, layer_num, similarity)

        out['batched_prompt'] = batched_prompt

        return out


    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean() * 1e-6


    def compute_conv_over_prompt(self, batched_prompt, f, layer_num, similarity):
        #  batch_size, dual, length // dual, self.num_heads, embed_dim // self.num_heads
        batched_prompt = batched_prompt.permute(1, 3, 0, 2, 4) #  dual, num_heads, B, length, head_dim
        k_prompt_list = []
        v_prompt_list = []
        
        k_prompt_layer = batched_prompt[0] #  num_heads, B,  length, head_dim
        v_prompt_layer = batched_prompt[1] # num_heads, B,length, head_dim
        n_heads, batch_size, length, head_dim = k_prompt_layer.shape

        length = length - self.ker_size + 1
        head_dim = head_dim - self.ker_size + 1
        new_k_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        for h in range(self.num_heads):
            k_conv_vals = self.k_conv_vals[str(layer_num)][str(h)]
            v_conv_vals = self.v_conv_vals[str(layer_num)][str(h)]
            k_prompt_head = k_prompt_layer[h].unsqueeze(1) #  B, 1, length, head_dim
            v_prompt_head = v_prompt_layer[h].unsqueeze(1) # B, 1, length, head_dim
            for p in range(f):
                k_conv_val = k_conv_vals[p]
                v_conv_val = v_conv_vals[p]

                new_k_prompt_layer[h] += k_conv_val(k_prompt_head).squeeze(1) * similarity[:, p].unsqueeze(1).unsqueeze(2)
                new_v_prompt_layer[h] += v_conv_val(v_prompt_head).squeeze(1)  * similarity[:, p].unsqueeze(1).unsqueeze(2)

        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0) # dual, num_heads, B, length, head_dim
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4) # B, dual, length, num_heads, head_dim


        return new_batched_prompt
    
    def compute_multi_prompt(self, batched_prompt, f, layer_num, similarity):

        B_size, pool_size, length_2, embed_dim = batched_prompt.shape

        pk = batched_prompt.reshape(B_size, 2, pool_size, int(length_2/2), self.num_heads, embed_dim // self.num_heads).permute(1, 4, 0, 2, 3, 5) 

        k_prompt_layer = pk[0] #  num_heads, B, pool, length, head_dim
        v_prompt_layer = pk[1] #  num_heads, B, pool, length, head_dim
        n_heads, batch_size, pool, length, head_dim = k_prompt_layer.shape

        new_k_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)
        new_v_prompt_layer = torch.zeros((n_heads, batch_size, length, head_dim), device=k_prompt_layer.device)

        for h in range(self.num_heads):
            k_prompt_head = k_prompt_layer[h] #  B, pool, length, head_dim
            v_prompt_head = v_prompt_layer[h] # B, pool, length, head_dim
            for p in range(2):
                new_k_prompt_layer[h] += k_prompt_head[:,p] * similarity[:, p].unsqueeze(1).unsqueeze(2)
                new_v_prompt_layer[h] += v_prompt_head[:,p] * similarity[:, p].unsqueeze(1).unsqueeze(2)

        new_batched_prompt = torch.stack([new_k_prompt_layer, new_v_prompt_layer], dim=0) # dual, num_heads, B, length, head_dim
        new_batched_prompt = new_batched_prompt.permute(2, 0, 3, 1, 4) # B, dual, length, num_heads, head_dim

        return new_batched_prompt
    

    def conv_ortho(self, weights):
        w = weights
        in_channels, out_channels, kernel_size, kernel_size = w.shape
        w =w.permute(1, 0, 2, 3). view(out_channels, -1)
        W1 = w.t()
        Ident = torch.eye(w.shape[1]).to(w.device)
        # print("W1 shape: ", W1.shape, w.shape)
        W_new = torch.matmul(W1, w)
        Norm = W_new - Ident
        b_k = torch.rand(Norm.shape[1]).to(Norm.device)
        b_k = b_k.unsqueeze(1)
        v1 = torch.matmul(Norm, b_k)
        norm1 = torch.sum(torch.square(v1))**0.5
        v2 = v1 / norm1
        v3 = torch.matmul(Norm, v2)

        return 0.01*(torch.sum(torch.square(v3))**0.5) + (1e-4)*(torch.sum(torch.square(w))**0.5)

    def conv_orthogonality(self, conv_vals):
        ortho_norm = 0
        for i in range(self.num_layers):

            for h in range(self.num_heads):

                conv_val = conv_vals[str(i)][str(h)]
                for j in range(len(conv_val)):
                    ortho_norm += self.conv_ortho(conv_val[j].weight)
        return ortho_norm
