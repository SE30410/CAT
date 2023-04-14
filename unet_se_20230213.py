from distutils.command.config import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import argparse
import copy
import numpy as np
from dse_duomotai_20230213 import SELayer2d_20230213 as att




class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''
    def __init__(self,config):
        super(Embeddings,self).__init__()
        img_size=config.img_size#120
        patch_size=config.patches#16

        n_patches=(img_size[0]//patch_size)*(img_size[1]//patch_size)

        self.patch_embeddings=nn.Conv2d(in_channels=config.in_channels_TR,
                                     out_channels=config.hidden_size,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        

        self.position_embeddings=nn.Parameter(torch.zeros(1,n_patches+1,config.hidden_size))
                                                          
                                                          
 
        self.classifer_token=nn.Parameter(torch.zeros(1,1,config.hidden_size))
        self.dropout=nn.Dropout((config.attention_dropout_rate))

    def forward(self,x):

        bs=x.shape[0]

        cls_tokens=self.classifer_token.expand(bs,-1,-1)#(bs,1,768)

        x=self.patch_embeddings(x)#

        x=x.flatten(2)#(bs,768,196)

        x=x.transpose(-1,-2)#(bs,196,768)
  
        x=torch.cat((cls_tokens,x),dim=1)#将分类信息与图片块进行拼接（bs,197,768）

        embeddings=x+self.position_embeddings#将图片块信息和对其位置信息进行相加(bs,197,768)

        embeddings=self.dropout(embeddings)

        return  embeddings



class Attention(nn.Module):
    def __init__(self,config):
        super(Attention,self).__init__()
        self.vis=config.vis
        self.num_attention_heads=config.num_heads#12-->8  
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64 128/12=10-->>128/8=16
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768      12*10=120-->

        self.query = nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key =nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value =nn.Linear(config.hidden_size, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout =nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):#

        mixed_query_layer = self.query(hidden_states)#wm,768->768

        mixed_key_layer = self.key(hidden_states)#wm,768->768

        mixed_value_layer = self.value(hidden_states)#wm,768->768


        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)

        key_layer = self.transpose_for_scores(mixed_key_layer)
  
        value_layer = self.transpose_for_scores(mixed_value_layer)

        
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
    
        attention_scores = attention_scores / np.math.sqrt(self.attention_head_size)#将结果除以向量维数的开方

        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率

        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        

        context_layer = torch.matmul(attention_probs, value_layer)#将概率与内容向量相乘
    
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
 
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,197)+(768,)=(bs,197,768)

        context_layer = context_layer.view(*new_context_layer_shape)

        
        attention_output = self.out(context_layer)
      
        attention_output = self.proj_dropout(attention_output)
      
        return attention_output, weights#wm,(bs,197,768),(bs,197,197)


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)#wm,786->3072
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)#wm,3072->786
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):#[vis=true]
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size#wm,768
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)#wm，层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h#残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h#残差结构
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vis = config.vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config,):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)#wm,对一幅图片进行切块编码，得到的是（bs,n_patch+1（196）,每一块的维度（768））
        self.encoder = Encoder(config)##

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)#wm,输出的是（bs,196,768)
        # print("6.Transformer_embedding_output=",embedding_output.shape)#[2, 197, 128]
        encoded, attn_weights = self.encoder(embedding_output)#wm,输入的是（bs,196,768)
        # print("6.Transformer_encoded=",encoded.shape)
        return encoded, attn_weights#输出的是（bs,197,768）



class VisionTransformer(nn.Module):#
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        self.num_classes = config.num_classes
        self.zero_head = config.zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config)
        self.head = nn.Linear(config.hidden_size, config.num_classes)#wm,768-->10

    def forward(self, x, labels=None):

        x, attn_weights = self.transformer(x)


        

        xa,_=x.split([30,1],dim=1)
  
        xa=xa.transpose(-1,-2)#

        xa=xa.reshape(x.size(0),32,18,15)



        logits = xa[:, :,:]



        if labels is not None:#没运行！！！
            print("do!!!")
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            # print("done!!!")
            return logits, attn_weights

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self,config):
    # def __init__(self,):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = 2
        self.bilinear =True
        std=4
        self.inc = DoubleConv(self.n_channels, std)
        self.down1 = Down(std, 2*std)
        self.down2 = Down(2*std, 4*std)
        self.down3 = Down(4*std,8*std)
        self.down4 = Down(8*std, 8*std)
        self.up1 = Up(16*std, 4*std, self.bilinear)
        self.up2 = Up(8*std, 2*std, self.bilinear)
        self.up3 = Up(4*std, std, self.bilinear)
        self.up4 = Up(2*std, std, self.bilinear)
        # self.outc = OutConv(64, self.n_classes)
        # self.Embeddings=Embeddings(config)
        self.VisionTransformer=VisionTransformer(config)##########
        self.se1_down4=att(32,32)
        self.se2_down3=att(16,16)
        self.se3_down2=att(8,8)
        self.se4_down1=att(4,4)

     
        

        self.fc1 = nn.Sequential(
            nn.Linear(int(std*120*144), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)  # 默认就是0.5
        )
        self.fc2= nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.fc3= nn.Sequential(
            nn.Linear(4096, self.n_classes)
        )
        self.fc_list = [self.fc1, self.fc2, self.fc3]
        # self.head = nn.Linear(config.hidden_size,2)#768-->2

        print("Unet_T Model Initialize Successfully!")

    def forward(self, CSF,GM,WM):

        #1.特征融合

        x=torch.cat([CSF,GM,WM],dim=1)
   
        #2.初步特征提




        #3.基础模块
        x1 = self.inc(x)
     
        x2 = self.down1(x1)

        x3 = self.down2(x2)
  
        x4 = self.down3(x3)


       
        

        x5 = self.down4(x4)


  
       
        #注意力模块
        x = self.up1(x5, self.se1_down4(x4))
        x = self.up2(x, self.se2_down3(x3))
        x = self.up3(x, self.se3_down2(x2))
        x = self.up4(x, self.se4_down1(x1))


     
        output = x.view(x.size()[0], -1) #[2, 4*224*224]
       
        for fc in self.fc_list:        # 3 FC
            output = fc(output)
       
        output=F.softmax(output)
        
        return output




