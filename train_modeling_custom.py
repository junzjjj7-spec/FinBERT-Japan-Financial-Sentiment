import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from huggingface_hub import login

# 📌 导入你自定义的模块 (确保 modeling_custom.py 在同一目录下)
from modeling_custom import CustomFinBERT, FocalLossTrainer


def main():
    # ==========================================
    # 1. 基础配置与检查
    # ==========================================
    # 检查 GPU
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"  # Mac M1/M2/M3 支持
        print("🚀 使用 Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        print("⚠️ 未检测到 GPU，将使用 CPU 训练 (速度较慢)")

    # 登录 Hugging Face (如果你在终端运行过 'huggingface-cli login' 则可注释此行)
    # 或者直接填入 Token
    # login(token="你的_WRITE_TOKEN_填在这里")

    # ==========================================
    # 2. 加载数据集
    # ==========================================
    # 建议使用你的 V5 Golden 终极版本
    dataset_id = "dgawghbuidw/finbert-japan-hybrid-v5-golden"
    print(f"\n📦 正在加载数据集: {dataset_id} ...")

    # 这一步需要联网
    dataset = load_dataset(dataset_id)
    print("数据集加载完成。")

    # ==========================================
    # 3. 数据预处理
    # ==========================================
    model_checkpoint = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    print("正在进行批量分词...")
    # num_proc=4 可以利用多核CPU加速处理
    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=1)

    # ==========================================
    # 4. 加载魔改模型
    # ==========================================
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    label2id = {"positive": 0, "negative": 1, "neutral": 2}

    print("正在初始化 CustomFinBERT (Mod 1 & 2)...")
    # 这里使用的是你 modeling_custom.py 里定义的类
    model = CustomFinBERT.from_pretrained(
        model_checkpoint,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    # ==========================================
    # 5. 定义指标函数
    # ==========================================
    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # 【修复】兼容 CustomModel 返回 tuple 的情况
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predictions = np.argmax(predictions, axis=1)
        acc = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # ==========================================
    # 6. 设置训练参数
    # ==========================================
    # 如果本地显存不够 (比如只有 8G)，请把 batch_size 改小，并增加 gradient_accumulation
    batch_size = 32
    # batch_size = 16 # 如果爆显存，解开这行

    experiment_name = "finbert-japan-advanced-final-local"

    args = TrainingArguments(
        output_dir=f"./output/{experiment_name}",

        # === 冠军参数 ===
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # === 本地运行特别配置 ===
        dataloader_num_workers=0,  # Windows下建议设为0，Linux可设为2或4
        logging_steps=50,  # 终端打印频率

        # === 策略 ===
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        # === 上传 ===
        push_to_hub=True,
        hub_model_id=f"dgawghbuidw/{experiment_name}",
        report_to="none"  # 本地跑通常关闭 wandb，除非你配置好了本地环境
    )

    # ==========================================
    # 7. 初始化魔改 Trainer (Mod 3)
    # ==========================================
    trainer = FocalLossTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # ==========================================
    # 8. 开始训练
    # ==========================================
    print("\n🔥 开始本地训练...")
    trainer.train()

    # ==========================================
    # 9. 最终测试与保存
    # ==========================================
    print("\n📝 正在进行最终测试集评估...")
    test_results = trainer.evaluate(tokenized_datasets["test"])

    print("=" * 40)
    print(f"🎯 最终测试集 F1 分数: {test_results['eval_f1']:.4f}")
    print(f"🎯 最终测试集 准确率: {test_results['eval_accuracy']:.4f}")
    print("=" * 40)

    # 保存到本地
    save_path = f"./saved_models/{experiment_name}"
    trainer.save_model(save_path)
    print(f"✅ 模型已保存到本地: {save_path}")

    # 推送到云端
    trainer.push_to_hub()
    print(f"🎉 模型已推送到 Hugging Face!")


if __name__ == "__main__":
    # 这一行是 Windows/Multiprocessing 运行所必须的
    main()