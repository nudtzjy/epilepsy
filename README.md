## Language / 语言
- [English](#english)
- [简体中文](#简体中文)

---

### English
1.config/dataConfig.yaml Some parameter information and data path about generating model training dataset are defined.

2.**epilepsy_svt_tfs.py** Training code of epilepsy detection model in the paper.  

3.**DataPreprocessLabel.py**  The data preprocessing code in the paper is used to generate the training data of the model

4.**epilepsy_svt_tfs_infer_args.py**   After the model training, use this code to load the code for model reasoning and prediction of the trained model. After the code execution, the detection results of different thresholds will be generated under the corresponding input file for subsequent detection effect analysis.

5.**model.py**   Some classes in the model are defined in the file.

6.**utils.py**  Traverse some files in the folder, read file content functions, and some sampling functions are defined in the file.

7.**requirements.txt**   Some environments and dependencies used in the project.


Load the script of reasoning based on the trained model in the paper, and the threshold value needs to be adjusted according to the model detection results. **Note**: considering that the original input file needs to be sliced according to the window length when reasoning, in order to speed up the calculation (reuse the sliced data to avoid re slicing the original data for each threshold), the threshold in the reasoning code is set by modifying the list in the code, and there is no external parameter transfer method for the time being.  

```
python epilepsy_svt_tfs_infer_args.py --threshold 0.6 --stride 5 --device cuda:0 --infer_ckpt data/ckpt/epoch_0_val_acc_0.997500000_model.pth --infer_input_path data/infer/PTX_CA3_U20130429_15_ch9-16_convert.mat
```

---

### 简体中文

1.config/dataConfig.yaml 里面定义了关于生成模型训练数据集的一些参数信息和数据路径。  

2.**epilepsy_svt_tfs.py** 是论文中癫痫检测模型的训练代码 。  

3.**DataPreprocessLabel.py**  是论文中的数据预处理代码，用于生成模型的训练数据。  

4.**epilepsy_svt_tfs_infer_args.py** 是模型训练完毕，使用该代码加载训练好的模型进行模型推理预测的代码，代码执行完毕会在对应的输入文件下面生成不同阈值的检测结果，用于后续的检测效果分析。  

5.**model.py** 模型中的一些类定义在文件中。  

6.**utils.py** 遍历文件夹中的一些文件、读取文件内容函数，以及一些采样函数定义在该文件中。 

7.**requirements.txt** 项目中使用到的一些环境和依赖。  


加载论文中训练好的模型进行推理的脚本,阈值需要根据模型检测结果进行调整。**注意**：考虑到推理的时候需要根据窗口长度对原始输入文件进行切片，为了加快计算（复用切片数据，避免跑每个阈值都要重新切片一次原始数据），推理代码中阈值的设置通过代码中修改列表的方式进行，暂时没有使用外部传参的方式。  

```
python epilepsy_svt_tfs_infer_args.py --threshold 0.6 --stride 5 --device cuda:0 --infer_ckpt data/ckpt/epoch_0_val_acc_0.997500000_model.pth --infer_input_path data/infer/PTX_CA3_U20130429_15_ch9-16_convert.mat
```

