# 九天编程小白的「ResNet→Transformer→ViT」奇幻漂流
从编程小白到完整实现三类深度学习模型，九天的学习严格围绕题目要求展开，每天聚焦“编程基础+模型原理”双维度突破，所有知识均服务于题目中“ResNet分类→Transformer Encoder→Vision Transformer”的核心任务，具体如下：


## 第一天：编程入门与任务认知
### 1. 编程基础：Python与PyTorch入门
- **Python核心语法**：从“变量定义、列表/字典”学起，掌握`for`循环、`if-else`判断（后续模型训练循环必备），理解函数定义（`def`关键字）与类的基础概念（`class`、`__init__`构造函数）——这是后续写`BasicBlock`、`SelfAttention`等类的前提；
- **PyTorch基础操作**：学会导入`torch`库，认识“张量（Tensor）”这一核心数据结构，掌握`torch.randn()`（生成随机输入，对应题目“输入随机矩阵”）、`tensor.shape`（查看维度，后续调试维度匹配必备）、`tensor.to(device)`（设备迁移，对应题目“GPU加速”）等基础操作。

### 2. 模型知识：ResNet核心概念认知
- 理解题目任务脉络：明确第一题需实现ResNet-18并完成CIFAR-10分类，初步认识“残差跳联”——知道普通卷积网络因深度增加会梯度消失，而ResNet的`F(x)+x`结构能让梯度直接通过恒等路径传递（对应题目的残差块示意图）；
- 记住ResNet-18的层级框架：对照题目表格，记下“conv1→maxpool→conv2_x~conv5_x→avgpool→fc”的流程，知道每类残差块（BasicBlock）含2个3×3卷积（后续第二天实现时无需再查结构）。


## 第二天：PyTorch神经网络组件与ResNet残差块实现
### 1. 编程基础：PyTorch神经网络核心组件
- **`nn.Module`基类**：学会继承`nn.Module`定义自定义网络层（如`class BasicBlock(nn.Module):`），理解`super().__init__()`的作用（初始化父类），掌握`forward`方法（前向传播逻辑，所有模型层的核心）；
- **卷积与批归一化层**：认识`nn.Conv2d`（参数：输入通道、输出通道、卷积核大小、步长、 padding）——对应题目中conv1的“7×7,64,stride2”；学会用`nn.BatchNorm2d`（批归一化，稳定训练，ResNet必用组件），知道`nn.ReLU`（激活函数，引入非线性）的`inplace=True`参数（节省内存）。

### 2. 模型知识：BasicBlock残差块实现逻辑
- 掌握残差块的“主路径+恒等路径”逻辑：主路径是“conv→bn→relu→conv→bn”，恒等路径是原始输入`x`；当输入输出通道/步长不匹配时（如conv3_x从64通道→128通道），需用`downsample`模块（1×1卷积+bn）统一维度（对应题目的残差块适配场景）；
- 调试维度匹配：学会用`assert`语句校验维度（如`assert dim_q == self.dim_q`），避免后续拼接/相加时报错——这是小白最易踩坑的点，提前通过简单案例（如输入[2,3,32,32]→conv2d后输出[2,64,16,16]）练习。


## 第三天：数据加载与ResNet完整架构
### 1. 编程基础：数据加载与训练逻辑
- **数据集与数据加载器**：学会用`torchvision.datasets.CIFAR10`下载数据集（对应题目“可通过torchvision下载”），用`transforms.Compose`组合预处理操作（随机裁剪`RandomCrop`、转张量`ToTensor`、标准化`Normalize`）——知道训练集需增强（提升泛化）、验证集仅标准化（保证评估客观）；
- **`DataLoader`使用**：掌握`torch.utils.data.DataLoader`的参数（`batch_size`批次大小、`shuffle=True`训练集打乱、`num_workers`多线程加载），解决CPU环境下`num_workers>0`报错的问题（设为0）；
- **训练循环基础**：理解“遍历DataLoader→取数据→前向传播→计算损失→反向传播→参数更新”的流程，认识`nn.CrossEntropyLoss`（分类任务损失函数）、`torch.optim.SGD`（优化器，带`momentum=0.9`提升稳定）。

### 2. 模型知识：ResNet-18完整架构与分类任务
- 完成ResNet-18组装：用`_make_layer`方法批量创建残差块组（conv2_x含2个BasicBlock、conv3_x含2个等，对应题目表格），添加全局平均池化`nn.AdaptiveAvgPool2d((1,1))`（动态适配输入尺寸，无需硬编码）和全连接层`nn.Linear(512,10)`（CIFAR-10为10类，对应题目分类任务）；
- 理解指标计算：学会用`torch.max(outputs,1)`取预测类别，通过`(predicted == labels).sum().item()`计算正确数，进而得到准确率（后续可视化指标必备）。


## 第四天：矩阵运算与Self Attention实现
### 1. 编程基础：PyTorch矩阵运算与掩码操作
- **批次矩阵乘法**：掌握`torch.bmm`（批量矩阵乘法，如Q[2,16,64] × K^T[2,64,16] → 注意力分数[2,16,16]），理解其与普通`torch.matmul`的区别（仅支持3维张量，适配批次数据）；
- **掩码处理**：学会用`tensor.masked_fill(mask == 0, -1e9)`屏蔽无效位置（对应题目“Mask (opt.)”），知道“将掩码位置设为极小值，SoftMax后权重接近0”的原理。

### 2. 模型知识：Scaled Dot-Product Attention原理
- 理解注意力的“全局信息提取”作用（对应题目“注意力机制能提取全局信息”）：Q（查询）、K（键）、V（值）均来自同一输入（Self Attention），通过计算Q与K的相似度得到权重，再加权V得到上下文向量；
- 掌握“缩放”的必要性：知道`1/math.sqrt(dim_k)`（`dim_k`为K的维度）能避免注意力分数过大，导致SoftMax梯度消失（对应题目“Scaled Dot-Product Attention”的“Scale”步骤）；
- 实现`SelfAttention`类：按“Linear映射Q/K/V→计算分数→缩放→掩码→SoftMax→加权V”的流程编码，确保输入[2,16,512]→输出[2,16,64]（维度匹配后续Multi-Head Attention）。


## 第五天：张量维度操作与Multi-Head Attention
### 1. 编程基础：张量维度进阶操作
- **维度转置与连续化**：学会`tensor.transpose(1,2)`（交换维度，如将[2,16,8,64]→[2,8,16,64]），理解`contiguous()`的作用——转置后张量内存离散，需调用该方法才能用`view`调整维度（小白最易忽略的坑，对应`_split_heads`方法）；
- **维度展平与拼接**：掌握`tensor.view()`（调整维度，如将[2,8,16,64]→[16,16,64]，展平批次+多头维度）、`torch.cat()`（拼接多头输出，如将8个[2,16,64]→[2,16,512]，对应题目“Concat”步骤）。

### 2. 模型知识：Multi-Head Attention的“多头机制”
- 理解“多头拆分”的意义：将输入拆分为多个子空间（如dim_model=512、num_heads=8→每个头dim=64），让模型关注不同维度的信息（对应题目“每个子空间分别执行Self Attention”）；
- 实现`MultiHeadAttention`类：完成“拆分多头（`_split_heads`）→子空间Self Attention→拼接多头（`_concat_heads`）→线性变换”的流程，验证输入[2,16,512]→输出[2,16,512]（维度与输入一致，适配后续Encoder）；
- 调试多头权重：学会计算所有头的平均权重（`att_weights_split.mean(dim=1)`），验证掩码有效性（屏蔽位置权重均值接近0，对应题目的掩码功能）。


## 第六天：层归一化与Transformer Encoder组件
### 1. 编程基础：层归一化与模块组装
- **层归一化**：认识`nn.LayerNorm`（对特征维度归一化，如[2,16,512]对最后一维归一化），理解其与`BatchNorm2d`的区别（LayerNorm不依赖批次，更适合序列数据，对应题目“Add&Norm”）；
- **Dropout正则化**：学会用`nn.Dropout(p=0.1)`防止过拟合，知道在训练时生效、验证时需用`model.eval()`关闭；
- **模块组合**：掌握用`nn.Sequential`包装多个层（如`downsample = nn.Sequential(conv, bn)`），理解“先定义组件、再在`forward`中调用”的模块化编程思路（如`EncoderLayer`中组合`multi_head_attn`、`add_norm1`等）。

### 2. 模型知识：EncoderLayer的“注意力+前馈”逻辑
- 理解Add&Norm层的作用：“Add”是残差连接（缓解梯度消失），“Norm”是层归一化（稳定训练，对应题目“Add&Norm”的核心意义），实现`AddNorm`类时加入可学习的`residual_weight`（增强灵活性）；
- 掌握FeedForward层结构：按“Linear→ReLU→Linear”编码（对应题目“Feed Forward”），隐藏层维度设为2048（Transformer常规配置）；
- 组装`EncoderLayer`：按“Multi-Head Attention→Dropout→Add&Norm→FeedForward→Dropout→Add&Norm”的顺序编写`forward`方法，确保每一步输入输出维度一致（均为[2,16,512]）。


## 第七天：层堆叠与Transformer Encoder验证
### 1. 编程基础：ModuleList与测试代码编写
- **层堆叠工具**：学会用`nn.ModuleList`堆叠多个`EncoderLayer`（如6层），理解其与`nn.Sequential`的区别（支持动态层数，适配不同Transformer版本）；
- **前向测试逻辑**：掌握“生成随机输入→初始化模型→调用`model(x)`→验证输出维度”的测试流程，学会用`torch.no_grad()`禁用梯度计算（节省内存，仅用于测试）；
- **日志打印**：用`print`输出输入/输出形状、权重均值等关键信息（如“输入形状：[2,16,512]，输出形状：[2,16,512]”），验证是否符合题目“输入随机矩阵看输出”的要求。

### 2. 模型知识：完整Transformer Encoder架构
- 理解Encoder的“层堆叠”逻辑：多个`EncoderLayer`串联，前一层的输出作为后一层的输入（对应题目“Nx”的含义），每一层都保留注意力权重（便于后续分析）；
- 满足题目输入要求：严格按“直接输入随机矩阵，不用Embedding”和“不做位置编码”实现，测试时生成`torch.randn(2,16,512)`作为输入，验证输出维度与输入一致，掩码后权重均值接近0（如第一批次第一序列后8位权重均值<0.0001）。


## 第八天：视觉转序列与Patch Embedding
### 1. 编程基础：图像张量与卷积分块
- **图像张量维度**：理解视觉数据的维度格式（[batch, channel, height, width]，如CIFAR-10为[2,3,32,32]），学会用`conv2d`实现“分块+投影”——通过`kernel_size=4`、`stride=4`，让3×32×32图像→768×8×8（对应4×4 Patch分块）；
- **维度转换实战**：掌握“卷积输出→展平→转置”的流程（`x.flatten(2)`将[2,768,8,8]→[2,768,64]，再`transpose(1,2)`→[2,64,768]），将图像转为Transformer可处理的序列格式。

### 2. 模型知识：ViT的“视觉→序列”迁移逻辑
- 理解ViT的核心思路：将图像拆分为Patch（对应题目“Patch Extraction”），每个Patch展平后线性投影为固定维度（对应“Linear Projection”），再按Transformer的序列输入处理（复用第二天的Encoder）；
- 计算Patch数量：学会`num_patches=(img_size//patch_size)**2`（如32//4=8→8×8=64个Patch），确保分块无重叠（`img_size%patch_size == 0`，通过`assert`校验）；
- 实现`PatchEmbedding`类：验证输入[2,3,32,32]→输出[2,64,768]，维度与后续Encoder输入（768）匹配（对应题目“复用二中完成的Encoder”）。


## 第九天：可学习参数与ViT完整实现
### 1. 编程基础：可学习参数与维度扩展
- **可学习参数定义**：学会用`nn.Parameter`定义可学习张量（如`class_token = nn.Parameter(torch.zeros(1,1,768))`），理解其与普通张量的区别（会被`model.parameters()`收录，参与梯度更新）；
- **维度扩展与拼接**：掌握`tensor.expand(batch_size, -1, -1)`（将[1,1,768]扩展为[2,1,768]，适配批次），用`torch.cat([class_token, patch_emb], dim=1)`拼接“分类Token+Patch序列”（得到[2,65,768]，65=64+1）。

### 2. 模型知识：ViT的完整前向流程
- 理解关键组件作用：[class] Token用于聚合全局特征（最终分类依赖其输出，对应题目“[class] embedding”），位置嵌入用于注入Patch的位置信息（可学习1D嵌入，对应“Positional Embedding”）；
- 实现`VisionTransformer`类：按“Patch Embedding→添加class Token→添加位置嵌入→Encoder编码→提取class Token输出→分类头”的流程编码，复用第七天的`TransformerEncoder`；
- 验证前向结果：加载CIFAR-10图像，执行前向传播，确保分类预测形状为[2,10]（CIFAR-10类别数）、注意力权重形状为[12,2,65,65]（12层Encoder），符合题目“实现ViT前向传播模型”的要求。


## 总结：小白到“模型实现者”的核心能力突破
九天的学习完全围绕题目需求展开，从“不会定义变量”到“能实现简单的ViT前向模型”，（虽然第一题绘图阶段Kernel总是崩溃）核心突破点在于：
1. **编程上**：掌握PyTorch核心组件（卷积、归一化、优化器）、张量维度操作（转置、拼接、展平）、数据加载与训练循环。

2. **模型上**：理解三类核心模型的设计逻辑——ResNet的残差跳联解决梯度消失、Transformer的注意力机制提取全局信息、ViT的“图像→序列”迁移思路，且所有理解均落地为符合题目要求的代码（如ResNet适配CIFAR-10、Transformer输入随机矩阵、ViT复用Encoder），完全呼应题目“循序渐进、相互耦合”的出题思路。
