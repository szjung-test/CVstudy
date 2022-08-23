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
```python
class Attention(nn.Module):
  def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0):
    seld.n_heads = n_heads
    self.dim = dim
    self.head_dim = dim // n_heads
    self.scale = self.head_dim ** -0.5 
    self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
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

## 3.MLP(Multi Layer Perceptron
```python
class MLP(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, p=0.):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = nn.GELU()
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop = nn.Dropout(p)
  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x
```

## 4. Transformer Encoder Block
```python
class Block(nn.Module):
  def __init__(self, dim, n_heads, mlp_ration=4.0, qkv_bias=True, p=0., attn_p=0.):
    super(Block, self).__init__()
    self.norm1 = nn.LayerNorm(dim, eps=1e-6)
    self.attn = Attention(
        dim,
        n_heads=n_heads,
        qkv_bias=qkv_bias,
        attn_p=attn_p
        proj_p=p
    )
    self.norm2 = nn.LayerNorm(dim, eps=1e-6)
    hidden_features = int(dim * mlp_ratio)
    self.mlp = MLP(
        in_features=dim,
        hidden_features=hidden_features,
        out_features=dim
    )
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

## 5. Vision Transformer
```python
class VisionTransformer(nn.Module):
  def __init__(
          self,
          img_size=384,
          patch_size=16,
          in_chans=3,
          n_classes=1000,
          embed_dim=768,
          depth=12,
          n_heads=12,
          mlp_ratio=4.
          qkv_bias=True,
          p=0.
          attn_p=0.):
     super(VisionTransformer, self).__init__()
     
     self.patch_embed = PatchEmbed(
          img_size=img_size,
          patch_size=patch_size,
          in_chans=in_chans,
          embed_dim=embed_dim,
          )
          self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
          self.pos_drop = nn.Dropout(p=p)
          
          self.blocks = nn.ModuleList(
            [
                Block(
                  dim=embed_dim,
                  n_heads=n_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  p=p,
                  attn_p=attn_p,
                )
                for _ in range(depth)
            ]
          )
          
          self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
          self.head = nn.Linear(embed_dim, n_classes)
          
      def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, 1+n_patches, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x
```

## 6. Model Output
```python
if __name__ == "__main__":
  from torchsummary import summary
  
  custom_config = {
    "img_size":384,
    "in_chans":3,
    "patch_size":16,
    "embed_dim":768,
    "depth":12,
    "n_heads":12,
    "qkv_bias":True,
    "mlp_ratio": 4
  }
  model_custom = VisionTransformer(**custom_config_
  
  inp = torch.rand(2, 3, 384, 384)
  res_c = model_custom(inp)
  print(res_c.shape)
  
  summary(model_custom, input_size=(3, 384, 384), device='cpu')
  print(model_custom)

```

## 7. Pretrained model inference
```python
import numpy as np
from PIL import Image
import torch.nn.fuctional
import cv2

k=10

imagenet_labels = dict(enumerate(open("classes.txt")))

model = torch.load("vit.path")
model.eval()

img = (np.array(Image.open("cat.jpg"))/128) - 1 # -1~1
img = cv2.resize(img, (384, 384))
inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits, dim=1)

top_probs, top_idxs = probs[0].topk(k)

for i, (idx_, prob_) in enumerate(zip(top_idxs, top_probs)):
  idx = idx_.item()
  prob = prob_.item()
  cls = imagenet_labels[idx].strip()
  print(f"{i}: {cls:<45} --- {prob:.4f}")

```
