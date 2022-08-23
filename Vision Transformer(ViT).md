# Model Architecture
1. 이미지를 여러개의 패치(base model의 patch는 16x16 크기)로 자른후에 각 패치별로 1차원 embedding demension(16x16x3 = 768) 으로 만든다.
2. class token이라는 것을 concatenate 시키고 각 패치마다 Position Embedding을 더해준다. (class token은 패치가 아닌 이미지 전체의 Embedding을 가지고 있다는 가정하에 최종 classification head에서 사용 / position embedding은 각 패치의 순서를 모델에 알려주는 역할을 한다) -> cls token과 positional embedding은 모두 학습되는 파라미터
3. Transformer Encoder를 n번 수행을 한다. (base model은 12번의 block 수행) -> Layer Normalization을 사용하며 기존 바닐라 Transformer와는 다르게 Attention과 MLP 이전에 수행하게 되면서 깊은 레이어에서도 학습ㅇ 잘 되도록 했다고 한다.
4. 최종적으로 Linear 연산을 통해 classification을 하게 된다.

## 1. Patch Embedding
```python
class PatchEmbed(nn.Module):
  def __init__(self, img_size, patch_sizem, in_chans=3, embed_dim=768:
    super(PatchEmbed, self).__init__()
    self.img_size = img_size
    self.patch_size = patch_size
    self.n_patches = (img_size // patch_size) ** 2
    
    self.proj = nn.Conv2d(
      in_channels=in_chans,
      out_channels=embed_dim,
      kernel_size=patch_size,
      stride=patch_size,
    ) # Embedding dim으로 변환하며 패치크기의 커널로 패치크기만큼 이동하며 이미지를 패치로 분할 할 수 있다.
    
  def forward(self, x):
    x = self.proj(x) # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
    x = x.flatten(2) # 세번째 차원부터 끝까지 flatten (batch_size, embed_dim, n_patches)
    x = x.transpose(1, 2) #(batch_size, n_patches, embed_dim)
    return x
````

## 2. Multi Head Attention
```
class Attention(nn.Module):
  def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0):
    seld.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.scale = self.head_dim ** -0.5 
    self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias
    self.atten_drop = nn.Dropout(attn_p)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_p)
    
  def forward(self, x):
  n_samples, n_tokens, dim = x.shape
  if dim != self.dim:
      raise ValueError
  
  qkv = self.qkv(x) #(n_samples, n_patches+1, dim*3)
  qkv = qkv.reshape(
    n_samples, n_tokens, 3, self.n_heads, self.head_dim
  )
  qkv = qkv.permute(2,0,3,1,4)
  q,k,v = qkv[0], qkv[1], qkv[2]
  k_t = k.transpose(-2, -1)
  
  dp =  (q @ k_t) * self.scale
  attn = dp.softmax(dim=-1)
  attn = self.attn_drop(attn)
  
  weighted_avg = attn @ v #(n_samples, n_heads, n_patches+1, head_dim)
  weighted_avg = weighted_avg.transpose(1,2)
  weighted_avg = weighted_avg.flatten(2)
  
  x = self.proj(weighted_avg)
  x = self.proj_drop(x)
  return x
```


