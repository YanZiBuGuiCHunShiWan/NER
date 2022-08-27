import torch.nn as nn

class IDCNN(nn.Module):

    def __init__(self,seqlen,embedd_dim,filters,kernel_size=3,num_blocks=4):
        super().__init__()
        self.seqlen=seqlen
        self.embedd_dim=embedd_dim
        self.filters=filters
        self.kernel_size=kernel_size
        self.num_blocks=num_blocks

        self.linear=nn.Linear(self.embedd_dim,filters)
        #定义单个block
        self.block=nn.Sequential()
        self.dilation_widths=[1,1,2] #论文设置
        for i in  range(len(self.dilation_widths)):
            dilation=self.dilation_widths[i]
            conv=nn.Conv1d(in_channels=self.filters,  #使用的是一维卷积
                           out_channels=self.filters,
                           kernel_size=self.kernel_size,
                           dilation=dilation,
                           padding=kernel_size // 2 + dilation - 1)
            self.block.add_module("layer%d"%i,conv)
            self.block.add_module("relu",nn.ReLU())
            self.block.add_module("layernorm",nn.LayerNorm(self.seqlen))

        #组合四个block
        self.idcnn=nn.Sequential()
        for i in range(num_blocks):
            self.idcnn.add_module("block %d"%i,self.block)
            self.idcnn.add_module("relu",nn.ReLU())
            self.idcnn.add_module("layernorm",nn.LayerNorm(self.seqlen))


    def forward(self,embeddings,lengths):
        """
        :param embeddings: [Batch_size,seqlen,embedd_dim]
        :param lengths:
        :return:
        """
        embeddings=self.linear(embeddings)  #[Batch_size,seqlen,embedd_dim]======>[Batch_size,seqlen,filters]
        embeddings=embeddings.permute(0,2,1) #因为一维卷积是在seqlen的方向上进行卷积操作，而pytorch的一维卷积是在最后两个维度进行操作的，因此我们需要先变换维度
        output=self.idcnn(embeddings).permute(0,2,1) #
        return output




