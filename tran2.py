# 1. 安装必要的库
!pip install transformers datasets evaluate accelerate seqeval wandb -q

# 2. 登录 Hugging Face (为了从你的仓库拉数据，以及把训练好的模型推回去)
from huggingface_hub import notebook_login
notebook_login()

# 3. 登录 WandB (为了看训练曲线，锻炼调参能力)
import wandb
wandb.login()
from datasets import load_dataset
# 2. 加载终极数据集 (直接加载，无需手动切分)
# ==========================================
dataset_id = "dgawghbuidw/finbert-japan-hybrid-v4"
print(f"📦 正在加载数据集: {dataset_id} ...")

raw_datasets = load_dataset(dataset_id)

# 打印一下看看结构，你应该能看到 train, validation, test 三个部分
print("数据集结构：", raw_datasets)
from transformers import AutoTokenizer

# 加载 FinBERT 分词器
model_checkpoint = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# === 关键修改 ===
# 定义映射关系 (告诉模型数字代表什么意思，这对以后推理很重要)
id2label = {0: "positive", 1: "negative", 2: "neutral"}
label2id = {"positive": 0, "negative": 1, "neutral": 2}


def preprocess_function(examples):
    # 1. 直接处理 'text' 列
    tokenized_inputs = tokenizer(examples["text"], truncation=True, max_length=128)

    # 2. 标签已经是数字了，什么都不用改！
    # Hugging Face 的 Trainer 会自动识别 'label' 列

    return tokenized_inputs


# 批量处理
print("正在进行分词处理...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# 检查一下处理后的结果 (确保 input_ids 都在，label 也是数字)
print("处理后样本：", tokenized_datasets['train'][0])
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification

# 加载评价指标
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    # 使用 weighted 平均，因为你的数据可能有类别不平衡
    f1_score = f1.compute(predictions=predictions, references=labels, average="weighted")

    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}


# === 加载模型 ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,  # 3分类
    id2label=id2label,  # 绑定映射关系
    label2id=label2id
)

print("模型加载成功，准备就绪！")
from transformers import TrainingArguments, Trainer

# 定义你的实验名称 (方便在 WandB 里看)
experiment_name = "finbert-japan-v1-lr2e5"

args = TrainingArguments(
    output_dir=f"./{experiment_name}",

    # === 核心调参区 ===
    learning_rate=2e-5,  # 尝试 2e-5, 3e-5, 5e-5
    per_device_train_batch_size=32,  # 尝试 16, 32
    per_device_eval_batch_size=32,
    num_train_epochs=3,  # 跑 5 轮，配合下面的早停
    weight_decay=0.01,  # 正则化，防止过拟合
    warmup_ratio=0.1,  # 热身步数比例

    # === 监控与保存 ===
    eval_strategy="epoch",  # 每个 epoch 测一次
    save_strategy="epoch",  # 每个 epoch 存一次
    load_best_model_at_end=True,  # 训练完自动加载最好的那个模型
    metric_for_best_model="f1",  # 谁的 F1 高，谁就是最好的
    report_to="wandb",  # 开启 WandB 监控图表
    run_name=experiment_name,  # WandB 里的名字

    # === 上传配置 ===
    push_to_hub=True,  # 训练完自动上传到你的 Hugging Face
    hub_model_id=f"dgawghbuidw/{experiment_name}",  # 你想上传的模型名字
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=args,
    # 重点在这里：直接指定数据集里对应的 split
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # 训练时只用 validation 做监控
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
print("🔥 开始最终训练...")
trainer.train()

# ==========================================
# 7. 【新增】最终测试集评估 (Final Evaluation)
# ==========================================
print("\n📝 正在进行最终大考 (Test Set Evaluation)...")
# 这里使用的是模型从未见过的 211 条纯真实数据
test_results = trainer.evaluate(tokenized_datasets["test"])

print("="*40)
print(f"🎯 最终测试集 F1 分数: {test_results['eval_f1']:.4f}")
print(f"🎯 最终测试集 准确率: {test_results['eval_accuracy']:.4f}")
print("="*40)