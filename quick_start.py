"""
HeteroGNN V3.0 - 快速开始脚本

基于门控元路径注意力的异构图神经网络
准确率: 97.88%
"""

import torch
from hetero_gnn_model_v3 import HeteroGNNRiskModel, create_model
from torch_geometric.data import HeteroData


def load_model(model_path='best_model_v3.pt', data_path='data/hetero_graph.pt'):
    """
    加载训练好的V3.0模型

    Args:
        model_path: 模型权重路径
        data_path: 图数据路径

    Returns:
        model: 加载好权重的模型
        data: 图数据
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    print(f"正在加载数据: {data_path}")
    data = torch.load(data_path)
    print(f"✓ 数据加载成功")
    print(f"  公司节点: {data['company'].num_nodes}")
    print(f"  行业节点: {data['industry'].num_nodes}")

    # 创建模型
    print(f"\n正在创建模型...")
    model = create_model(
        data=data,
        hidden_channels=256,
        num_layers=3,
        num_heads=4,
        dropout=0.4
    ).to(device)

    # 加载权重
    print(f"正在加载模型权重: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 模型加载成功")
    print(f"  训练轮数: {checkpoint['epoch'] + 1}")
    print(f"  准确率: {checkpoint['acc']*100:.2f}%")
    print(f"  F1-Macro: {checkpoint['f1_macro']*100:.2f}%")

    return model, data, device


def predict(model, data, device):
    """
    使用模型进行预测

    Args:
        model: 训练好的模型
        data: 图数据
        device: 计算设备

    Returns:
        predictions: 预测类别
        probabilities: 预测概率
    """
    print(f"\n正在预测...")

    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

        output = model(x_dict, edge_index_dict)

        predictions = output.argmax(dim=1)
        probabilities = torch.softmax(output, dim=1)

    print(f"✓ 预测完成")

    return predictions, probabilities


def analyze_gates(model, data, device):
    """
    分析门控值

    Args:
        model: 训练好的模型
        data: 图数据
        device: 计算设备

    Returns:
        gate_stats: 门控值统计信息
    """
    print(f"\n正在分析门控值...")

    model.eval()
    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

        _ = model(x_dict, edge_index_dict)

    gate_stats = model.get_metapath_attention_weights()

    if gate_stats:
        print(f"✓ 门控值分析完成")
        print(f"\n门控值统计:")
        print(f"  均值: {gate_stats['mean']:.4f}")
        print(f"  标准差: {gate_stats['std']:.4f}")
        print(f"  最小值: {gate_stats['min']:.4f}")
        print(f"  最大值: {gate_stats['max']:.4f}")
        print(f"  中位数: {gate_stats['median']:.4f}")
    else:
        print(f"✗ 无法获取门控值统计")

    return gate_stats


def main():
    """主函数"""
    print("=" * 80)
    print("HeteroGNN V3.0 - 快速开始")
    print("=" * 80)
    print(f"\n性能指标:")
    print(f"  准确率: 97.88%")
    print(f"  F1-Macro: 98.11%")
    print(f"  所有类别F1 > 97%")

    # 加载模型
    model, data, device = load_model(
        model_path='best_model_v3.pt',
        data_path='data/hetero_graph.pt'
    )

    # 预测
    predictions, probabilities = predict(model, data, device)

    # 显示前10个预测结果
    print(f"\n前10个预测结果:")
    print(f"{'节点ID':<10} {'预测类别':<10} {'置信度'}")
    print("-" * 40)
    for i in range(min(10, len(predictions))):
        pred_class = predictions[i].item()
        confidence = probabilities[i, pred_class].item()
        print(f"{i:<10} {pred_class:<10} {confidence:.4f}")

    # 分析门控值
    gate_stats = analyze_gates(model, data, device)

    # 保存预测结果
    results = {
        'predictions': predictions.cpu().numpy().tolist(),
        'probabilities': probabilities.cpu().numpy().tolist(),
        'gate_statistics': gate_stats
    }

    import json
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 预测结果已保存到: prediction_results.json")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
