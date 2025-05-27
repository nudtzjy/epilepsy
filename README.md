1.config/dataConfig.yaml 里面定义了关于生成模型训练数据集的一些参数信息和数据路径。  

2.**epilepsy_svt_tfs.py** 是论文中癫痫检测模型的训练代码 。  

3.**DataPreprocessLabel.py**  是论文中的数据预处理代码，用于生成模型的训练数据。  

4.**epilepsy_svt_tfs_infer_args.py** 是模型训练完毕，使用该代码加载训练好的模型进行模型推理预测的代码，代码执行完毕会在对应的输入文件下面生成不同阈值的检测结果，用于后续的检测效果分析。  

5.**model.py** 模型中的一些类定义在文件中。  

6.**utils.py** 遍历文件夹中的一些文件、读取文件内容函数，以及一些采样函数定义在该文件中。 

7.**requirements.txt** 项目中使用到的一些环境和依赖。  


加载论文中训练好的模型进行推理的脚本,阈值需要根据模型检测结果进行调整。  

```
python epilepsy_svt_tfs_infer_args.py --threshold 0.6 --stride 5 --device cuda:0 --infer_ckpt data/ckpt/epoch_0_val_acc_0.997500000_model.pth --infer_input_path data/infer/PTX_CA3_U20130429_15_ch9-16_convert.mat
```

