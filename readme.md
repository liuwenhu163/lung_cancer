简单列一下相关环境配置
第一步安装medpy库：pip install medpy -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
第二步安装 pylic==0.2.1
第三步安装pytorch 推荐pytorch版本为1.8.1

最后pip install -r requriments.txt

简单介绍一下目前的网络配置
先用3x3的二维卷积搭配cbam注意力模块提取图像的纹理信息，颜色信息
其次用双向convlstm模块提取序列数据的关联特征，这部分起到联系上下文图像信息的作用
最后搭配一个短期矫正金字塔金字塔，将双向convlstm提取到的上下帧图像信息进行多尺度矫正
然后通过ASPP模块进行多尺度特征融合（待加入）
添加残差块，得到最终的分割输出图

模型图：（待加入）
