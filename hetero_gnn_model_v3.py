"""
基于溢出网络的新能源汽车风险异构图神经网络风险识别模型
包含多种注意力机制：
1. 行业特征注意力机制
2. 企业特征注意力机制
3. 门控元路径注意力机制（改进版）
4. 企业-行业注意力机制

V3版本：使用门控机制改进元路径注意力，提升参数效率和表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from typing import Dict, Optional


class CompanyFeatureAttention(nn.Module):
    """企业特征注意力机制"""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super(CompanyFeatureAttention, self).__init__()
        projected_dim = ((in_channels // num_heads) * num_heads)
        if projected_dim == 0:
            projected_dim = num_heads

        self.proj = nn.Linear(in_channels, projected_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=projected_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(projected_dim)
        self.ffn = nn.Sequential(
            nn.Linear(projected_dim, projected_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projected_dim * 4, out_channels)
        )

    def forward(self, x):
        x_proj = self.proj(x)
        x_seq = x_proj.unsqueeze(1)
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.squeeze(1)
        x = self.norm(x_proj + attn_output)
        out = self.ffn(x)
        return out


class IndustryFeatureAttention(nn.Module):
    """行业特征注意力机制"""

    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 4):
        super(IndustryFeatureAttention, self).__init__()
        projected_dim = ((in_channels // num_heads) * num_heads)
        if projected_dim == 0:
            projected_dim = num_heads

        self.proj = nn.Linear(in_channels, projected_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=projected_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(projected_dim)
        self.ffn = nn.Sequential(
            nn.Linear(projected_dim, projected_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projected_dim * 4, out_channels)
        )

    def forward(self, x):
        x_proj = self.proj(x)
        x_seq = x_proj.unsqueeze(1)
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        attn_output = attn_output.squeeze(1)
        x = self.norm(x_proj + attn_output)
        out = self.ffn(x)
        return out


class CompanyIndustryAttention(nn.Module):
    """企业-行业注意力机制"""

    def __init__(self, company_channels: int, industry_channels: int, out_channels: int, num_heads: int = 4):
        super(CompanyIndustryAttention, self).__init__()
        self.company_proj = nn.Linear(company_channels, out_channels)
        self.industry_proj = nn.Linear(industry_channels, out_channels)
        actual_heads = min(num_heads, out_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=actual_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, company_x, industry_x, edge_index):
        company_h = self.company_proj(company_x)
        industry_h = self.industry_proj(industry_x)
        source, target = edge_index
        queries = company_h[source]
        keys = industry_h[target]
        values = industry_h[target]
        queries = queries.unsqueeze(0)
        keys = keys.unsqueeze(0)
        values = values.unsqueeze(0)
        attn_output, _ = self.cross_attn(queries, keys, values)
        attn_output = attn_output.squeeze(0)
        company_out = torch.zeros_like(company_h)
        company_out.index_add_(0, source, attn_output)
        edge_counts = torch.zeros(company_h.size(0), device=company_h.device)
        edge_counts.index_add_(0, source, torch.ones(source.size(0), device=company_h.device))
        company_out = company_out / (edge_counts.unsqueeze(1) + 1e-6)
        company_out = self.norm(company_h + company_out)
        return company_out


class GatedMetapathAttention(nn.Module):
    """
    门控元路径注意力机制

    核心改进：
    1. 使用门控网络自适应调整每个元路径的贡献
    2. 元路径特定的变换保留语义差异
    3. 共享投影层减少参数量
    4. 残差连接稳定训练
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4):
        super(GatedMetapathAttention, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_metapaths = 3  # direct, industry_mediated, supply_chain

        # 共享投影层（所有元路径共享）
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )

        # 元路径特定的变换（保留语义差异）
        self.metapath_transforms = nn.ModuleDict({
            'direct': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            ),
            'industry_mediated': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            ),
            'supply_chain': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels)
            ),
        })

        # 门控网络
        # 输入：拼接的元路径嵌入 [N, 3*D]
        # 输出：门控值 [N, D]
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()  # 输出0-1之间的门控值
        )

        # 输出归一化
        self.output_norm = nn.LayerNorm(hidden_channels)

        # 用于可视化
        self.last_gate_values = None

    def forward(self, metapath_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            metapath_embeddings: {
                'direct': [N, D],
                'industry_mediated': [N, D],
                'supply_chain': [N, D]
            }

        Returns:
            聚合后的嵌入 [N, D]
        """
        metapath_names = ['direct', 'industry_mediated', 'supply_chain']

        # 收集并变换每个元路径
        transformed_embeddings = []
        raw_embeddings = []

        for name in metapath_names:
            if name in metapath_embeddings and metapath_embeddings[name] is not None:
                # 元路径特定的变换
                h_transformed = self.metapath_transforms[name](metapath_embeddings[name])

                # 共享投影
                h_projected = self.shared_projection(h_transformed)

                transformed_embeddings.append(h_projected)
                raw_embeddings.append(metapath_embeddings[name])

        if not transformed_embeddings:
            return None

        # 堆叠 [3, N, D]
        stacked = torch.stack(transformed_embeddings, dim=0)

        # 计算门控值
        # 拼接所有元路径的原始嵌入 [N, 3*D]
        concat_raw = torch.cat(raw_embeddings, dim=-1)

        # 计算门控值 [N, D]
        gate_values = self.gate_network(concat_raw)

        # 保存门控值用于可视化
        self.last_gate_values = gate_values  # [N, D]

        # 应用门控并聚合
        # stacked: [3, N, D]
        # gate_values: [N, D]
        # 广播乘法: gate_values.unsqueeze(0) [1, N, D]
        gated_output = (stacked * gate_values.unsqueeze(0)).sum(dim=0)  # [N, D]

        # 残差连接（使用direct路径作为基准）
        baseline = metapath_embeddings.get('direct', raw_embeddings[0])
        output = self.output_norm(gated_output + baseline)

        return output

    def get_gate_statistics(self):
        """获取门控值的统计信息"""
        if self.last_gate_values is None:
            return None

        gates = self.last_gate_values  # [N, D]

        return {
            'mean': gates.mean().item(),
            'std': gates.std().item(),
            'min': gates.min().item(),
            'max': gates.max().item(),
            'median': gates.median().item()
        }


class MetapathAggregator(nn.Module):
    """元路径特定的消息聚合器"""

    def __init__(self, hidden_channels: int, num_heads: int, dropout: float):
        super(MetapathAggregator, self).__init__()

        self.conv = HeteroConv({
            ('company', 'spillover', 'company'): GATConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=False
            ),
            ('company', 'belongs_to', 'industry'): GATConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=False
            ),
            ('industry', 'contains', 'company'): GATConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=False
            ),
            ('industry', 'supply_chain', 'industry'): GATConv(
                hidden_channels, hidden_channels // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=False
            ),
        }, aggr='mean')

        self.norm_company = nn.LayerNorm(hidden_channels)
        self.norm_industry = nn.LayerNorm(hidden_channels)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[tuple, torch.Tensor],
                metapath_type: str) -> Dict[str, torch.Tensor]:

        company_residual = x_dict['company']
        industry_residual = x_dict.get('industry', None)

        if metapath_type == 'direct':
            selected_edge_types = [('company', 'spillover', 'company')]
        elif metapath_type == 'industry_mediated':
            selected_edge_types = [
                ('company', 'belongs_to', 'industry'),
                ('industry', 'contains', 'company')
            ]
        elif metapath_type == 'supply_chain':
            selected_edge_types = [
                ('company', 'spillover', 'company'),
                ('company', 'belongs_to', 'industry'),
                ('industry', 'contains', 'company'),
                ('industry', 'supply_chain', 'industry')
            ]
        else:
            raise ValueError(f"Unknown metapath type: {metapath_type}")

        selected_edge_index = {}
        for edge_type in selected_edge_types:
            if edge_type in edge_index_dict:
                selected_edge_index[edge_type] = edge_index_dict[edge_type]

        out_dict = self.conv(x_dict, selected_edge_index)

        if 'company' in out_dict:
            out_dict['company'] = self.norm_company(company_residual + out_dict['company'])
        else:
            out_dict['company'] = company_residual

        if 'industry' in out_dict and industry_residual is not None:
            out_dict['industry'] = self.norm_industry(industry_residual + out_dict['industry'])
        elif industry_residual is not None:
            out_dict['industry'] = industry_residual

        return out_dict


class HeteroGNNRiskModel(nn.Module):
    """
    异构图神经网络风险识别模型 - V3版本

    关键改进：
    1. 使用门控元路径注意力
    2. 更好的参数效率
    3. 更强的表达能力
    """

    def __init__(
        self,
        company_channels: int,
        industry_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4
    ):
        super(HeteroGNNRiskModel, self).__init__()

        self.company_channels = company_channels
        self.industry_channels = industry_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        # 1. 特征注意力
        self.company_feature_attn = CompanyFeatureAttention(
            company_channels, hidden_channels, num_heads
        )

        self.industry_feature_attn = IndustryFeatureAttention(
            industry_channels, hidden_channels, num_heads
        )

        # 2. 企业-行业注意力
        self.company_industry_attn = CompanyIndustryAttention(
            hidden_channels, hidden_channels, hidden_channels, num_heads
        )

        # 3. 元路径聚合器
        self.metapath_aggregators = nn.ModuleList([
            MetapathAggregator(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 4. 门控元路径注意力（改进版）
        self.metapath_attn = GatedMetapathAttention(hidden_channels, num_heads)

        # 5. 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # 1. 特征注意力
        company_x = self.company_feature_attn(x_dict['company'])
        industry_x = self.industry_feature_attn(x_dict['industry'])

        # 2. 企业-行业交叉注意力
        if ('company', 'belongs_to', 'industry') in edge_index_dict:
            company_x = self.company_industry_attn(
                company_x,
                industry_x,
                edge_index_dict[('company', 'belongs_to', 'industry')]
            )

        # 3. 元路径特定的消息传递
        metapath_embeddings = {}

        for metapath in ['direct', 'industry_mediated', 'supply_chain']:
            x_dict_mp = {
                'company': company_x.clone(),
                'industry': industry_x.clone()
            }

            for aggregator in self.metapath_aggregators:
                x_dict_mp = aggregator(x_dict_mp, edge_index_dict, metapath_type=metapath)
                x_dict_mp['company'] = F.relu(x_dict_mp['company'])

            metapath_embeddings[metapath] = x_dict_mp['company']

        # 4. 门控元路径注意力聚合
        final_emb = self.metapath_attn(metapath_embeddings)

        if final_emb is None:
            final_emb = metapath_embeddings['direct']

        # 5. 分类
        out = self.classifier(final_emb)

        return out

    def predict(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """预测"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict, edge_attr_dict)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        return preds, probs

    def get_metapath_attention_weights(self):
        """获取门控值的统计信息"""
        return self.metapath_attn.get_gate_statistics()

    def get_gate_values(self):
        """获取原始门控值"""
        return self.metapath_attn.last_gate_values


def create_model(data: HeteroData, hidden_channels: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.3) -> HeteroGNNRiskModel:
    """创建模型"""
    company_channels = data['company'].x.size(1)
    industry_channels = data['industry'].x.size(1)
    num_classes = int(data['company'].y.max().item() + 1)

    model = HeteroGNNRiskModel(
        company_channels=company_channels,
        industry_channels=industry_channels,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads
    )

    return model


if __name__ == '__main__':
    print("测试V3版本的模型（门控元路径注意力）...")
    from torch_geometric.data import HeteroData

    num_companies = 100
    num_industries = 20
    company_channels = 10
    industry_channels = 24
    hidden_channels = 128
    num_classes = 4

    data = HeteroData()
    data['company'].x = torch.randn(num_companies, company_channels)
    data['company'].y = torch.randint(0, num_classes, (num_companies,))
    data['industry'].x = torch.randn(num_industries, industry_channels)

    data['company', 'spillover', 'company'].edge_index = torch.randint(
        0, num_companies, (2, 200)
    )
    data['company', 'belongs_to', 'industry'].edge_index = torch.stack([
        torch.randint(0, num_companies, (150,)),
        torch.randint(0, num_industries, (150,))
    ])
    data['industry', 'contains', 'company'].edge_index = torch.stack([
        torch.randint(0, num_industries, (150,)),
        torch.randint(0, num_companies, (150,))
    ])
    data['industry', 'supply_chain', 'industry'].edge_index = torch.randint(
        0, num_industries, (2, 50)
    )

    model = create_model(data, hidden_channels=hidden_channels, num_layers=2,
                        num_heads=2, dropout=0.3)

    print("\n模型结构:")
    print(model)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")

    print("\n测试前向传播...")
    model.train()
    output = model(data.x_dict, data.edge_index_dict)
    print(f"✓ 输出形状: {output.shape}")

    preds, probs = model.predict(data.x_dict, data.edge_index_dict)
    print(f"✓ 预测形状: {preds.shape}")
    print(f"✓ 概率形状: {probs.shape}")

    # 测试门控值获取
    model.train()
    _ = model(data.x_dict, data.edge_index_dict)
    gate_stats = model.get_metapath_attention_weights()
    if gate_stats:
        print(f"\n门控值统计:")
        print(f"  均值: {gate_stats['mean']:.4f}")
        print(f"  标准差: {gate_stats['std']:.4f}")
        print(f"  范围: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")

    print("\n✓ 模型测试成功！")
