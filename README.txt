Jupyter入口：Jupyter五折训练与结果查看.ipynb。
使用说明最前面有逐单元格操作步骤。训练程序默认执行全部五折；fold0是第1折的内部编号。
五折指标表：cv_metrics.csv；均值及标准差：cv_summary.csv。

新增：完整训练及性能评估入口为 scripts/train_and_evaluate.py，详见 训练与评估使用说明.txt。
默认只做训练内验证，显式 --final-epochs 才执行最终重训和独立测试。

第18组：LKA 数据预处理与模型构建

入口：reports/数据处理与模型构建说明.html
可直接编辑的报告正文：reports/初步报告第1_2部分_可编辑.txt

已完成
1. 全部原始 CSV 与道路帧的审计，帧级标签、配对表与固定分组划分。
2. 110维统计特征，100点18通道原始序列，18个有效掩码通道，6维质量信息。
3. 每个交叉验证训练折独立拟合填补和标准化参数；测试使用全训练集参数。
4. XGBoost 基线及 MLP、TCN、CAN双路径、图像、基础融合、门控模型。
5. 真实样本前向/反向与数据不泄漏检查；未调参XGBoost的五折特征对照。

关键文件
src/lka/config.py                  输入字段和窗口参数
src/lka/preprocessing.py           因果重采样、统计特征、训练折变换
src/lka/data.py                    PyTorch 数据加载器
src/lka/models.py                  深度模型定义
scripts/prepare_data.py            从原始CSV/图像重新生成全部数据
scripts/check_models.py            真实批次与因果/泄漏/门控检查
scripts/check_xgboost.py           仅训练内的五折基线验证
scripts/build_materials.py         生成统计图和报告材料
artifacts/manifest.csv             每帧图像/CSV、标签、关联组、fold清单
artifacts/features_raw.npz         stats、sequence、quality、label
artifacts/transform_fold*.npz      每折训练集拟合的参数
artifacts/transform_train_all.npz  最终全训练集拟合参数
artifacts/audit_files.json         逐CSV审计（包括被排除文件）
artifacts/variable_audit.csv        所选原始变量的缺失/范围检查
artifacts/feature_dictionary.csv   110个特征名称
artifacts/duplicate_frames.csv     精确相同图像及其分组
artifacts/model_checks.json        模型结构检查结果，不是训练性能
artifacts/xgboost_validation.csv   五折验证结果，未评分独立测试集

数据说明
原始数据已解压到工作区 data/raw/TESLA_AP3_MODEL3_v1，不在本项目内重复保存。
未复制或修改原始CSV/图像；标签均使用原文件全部有效 LKA_error 的有符号均值。
默认采用 can_NNNN.csv 与 frame_NNNN.jpg 后缀配对，CAN最大Time为候选端点。
图像没有独立采集时间戳，端点与曝光时刻需向数据提供者确认。
不对负误差取绝对值。115帧均值小于-0.25，按原题仍为正常，应确认该定义。
官方Train/Test不重新随机打散；完全相同图像关联记录合并分组。
没有足够信息保证按完整驾驶行程分组；当前完成记录/重叠/完全重复检查。
特征原始数组保留NaN；数据加载器在指定训练折参数下填补与标准化。
均值基线为18维（14个连续/派生均值+4个状态比例）。

历史充分子集
491份CSV只有一个采样点，不把它们伪造成完整历史。默认数据加载器过滤
history_eligible：时间跨度至少1.98秒且可用比例至少90%，共2333帧，
训练1972帧、测试361帧。其余549帧保留作后续缺失实验。569帧不足完整2秒，
其中部分覆盖1.98秒网格，因此与549的数量不同。
默认变换参数只在各训练折的历史充分样本上拟合。require_history=False可读取
全部配对样本，仍复用训练部分已拟合的参数；请单独声明该实验人群。
完整样本结果与历史充分子集结果不能混为同一个实验。

当前机器复现
在项目文件夹运行 ./run_checks.sh。
主Python：/opt/anaconda3/bin/python（PyTorch/torchvision/sklearn）
基线Python：/opt/anaconda3/envs/pytorch_env/bin/python（XGBoost/sklearn）
run_checks.sh 不重新解压或重做预处理，可加 --rebuild 从原始数据重建。
requirements.txt列出统一环境所需包；本机为了避免下载与环境改动使用两个已有环境。

代码使用示例（在主Python下运行）
import sys
sys.path.insert(0, 'src')
from lka.data import LKADataset
from lka.models import LKAModel
from torch.utils.data import DataLoader
batch = next(iter(DataLoader(LKADataset('train', fold=0), batch_size=4)))
model = LKAModel(110, 36, 6, mode='gated')
logits, weights = model(batch, return_gate=True)
# 交叉熵使用logits；不要先softmax。
# 验证用LKADataset('val', fold=0)，最终测试用LKADataset('test')。
# 最终测试前须用LKADataset('train_all')和全训练参数重新训练最终模型。

报告PDF：reports/初步报告第1_2部分_数据处理与模型构建.pdf（3页，正文12号）
结构图：reports/architecture.svg（矢量，可插入报告或PPT）

共享给组员
压缩包包含处理后的特征和代码，不含2.7GB原始数据。组员需将同一数据集解压到
lka_project同级的data/raw/TESLA_AP3_MODEL3_v1；也可修改config.py中的RAW和
WORKSPACE并重新生成样本索引。已有manifest中的image_path是相对工作区的路径。

下一步属于训练与评估工作
确认图像时间端点；决定并记录本地预训练权重来源；AdamW/交叉熵完整训练；
在验证集调参、早停和阈值选择；模态扰动训练与鲁棒性对照；最终一次独立测试。
当前ResNet检查为随机初始化。训练时可向LKAModel传weights_path加载本地ResNet-18
原始state_dict；默认不隐式联网，不声称已完成预训练或性能验证。

缺失模态接口
本次数据加载器仅返回有效配对样本，modality_mask均为True。
模型已支持单模态缺失掩码，双模态缺失拒绝预测。
做鲁棒性实验时，除改mask，还要一致更新图像/序列/统计特征及quality；
CAN序列受损后要重新生成统计量与掩码，不能只改一条分支。

完整实验尚未进行，不能将model_checks中的随机网络损失写作模型性能。
XGBoost的AP为average_precision_score，不是梯形积分计算的PR曲线面积。
