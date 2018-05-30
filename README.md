# facenet-demo
## 概述
  基于GOOGLE FACENET的人脸识别演示程序，开发环境为python3.6+tensorflow
  - 执行人脸采集
 ```
 python3 catchFromCamera.py
 ```
 - 训练分类器
 ```
 python3 classifier.py
 ```
 - 实时人脸识别（从电脑摄像头）
 ```
 python3 recognitionFromCamera.py
 ```
 - 录制视频文件
 ```
 python3 recordVedio.py
 ```
 - 从一段视频（某个人的单独视频）中检查人脸并训练分类器，从另一段视频中识别出前一段视频中的人物。
 ```
 python3 demo.py
 ```
## 环境部署
- 安装python3.6
  这个网上有各种教程，这里就不赘述了。
- 安装Tensorflow
- 预训练模型
  使用了GOOGLE官方的预训练模型，下载地址：

  Pre-trained models
  <table>
    <tr>
    <td>
      Model name</td>
      <td>LFW accuracy</td>
      <td>Training dataset</td>
      <td>Architecture</td>
    </tr>
    <tr>
      <td><a href="https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz">20180408-102900</a></td>
      <td>0.9905</td>
      <td>CASIA-WebFace</td>
      <td>Inception ResNet v1</td>
    </tr>
    <tr>
      <td><a href="https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz">20180402-114759</a></td>
      <td>0.9965</td>
      <td>VGGFace2</td>
      <td>Inception ResNet v1</td>
    </tr>
  </table>
  预训练模型下载后解压至models目录，然后修改face.py中的模型路径即可

## 需要完善的功能
- 目前分类训练每次都要重新训练所有的图片，需要增加增量训练分类器的功能；
- 需要测试Tensorflow-GPU,利用CUDA来实现性能提升；
- 针对少量图片来进行训练，例如只提供两到三张图片，拟引入KERAS的图片增强功能来扩充训练样本数量；
- 测试在复杂条件下的准确率

## 引用资源
 大部分代码引用了https://github.com/davidsandberg/facenet 代码


 
