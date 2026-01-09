# -*- coding: utf-8 -*-
"""
RAG System Quality Metrics
质量评估指标模块
"""

import numpy as np
from typing import List, Dict, Any
import re

class QualityEvaluator:
    """质量评估器"""

    def __init__(self, embedding_model):
        """
        初始化评估器

        Args:
            embedding_model: 嵌入模型，用于计算相似度
        """
        self.embedding_model = embedding_model

    def evaluate_generation_quality(self, query: str, answer: str, context: str = None) -> Dict[str, float]:
        """
        评估生成质量

        Args:
            query: 问题
            answer: 答案
            context: 上下文（可选）

        Returns:
            质量指标字典
        """
        metrics = {}

        # 1. 答案长度
        metrics['answer_length'] = len(answer)

        # 2. 答案相关性（使用嵌入模型计算）
        metrics['relevance'] = self._compute_relevance(query, answer)

        # 3. 答案完整性（基于上下文）
        if context:
            metrics['completeness'] = self._compute_completeness(answer, context)
        else:
            metrics['completeness'] = 0.0

        # 4. 答案流畅性（基于句子结构）
        metrics['fluency'] = self._compute_fluency(answer)

        # 5. 答案信息密度
        metrics['information_density'] = self._compute_information_density(answer)

        return metrics

    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[str],
                                   distances: List[float]) -> Dict[str, float]:
        """
        评估检索质量

        Args:
            query: 问题
            retrieved_docs: 检索到的文档列表
            distances: 距离列表（相似度得分）

        Returns:
            检索质量指标字典
        """
        metrics = {}

        # 1. 平均相似度
        if distances:
            # 将距离转换为相似度（余弦距离 -> 相似度）
            similarities = [1 - d for d in distances if d <= 1.0]
            if similarities:
                metrics['avg_similarity'] = np.mean(similarities)
                metrics['max_similarity'] = np.max(similarities)
                metrics['min_similarity'] = np.min(similarities)
            else:
                metrics['avg_similarity'] = 0.0
                metrics['max_similarity'] = 0.0
                metrics['min_similarity'] = 0.0
        else:
            metrics['avg_similarity'] = 0.0
            metrics['max_similarity'] = 0.0
            metrics['min_similarity'] = 0.0

        # 2. 检索多样性（文档之间的差异）
        if len(retrieved_docs) > 1:
            metrics['diversity'] = self._compute_diversity(retrieved_docs)
        else:
            metrics['diversity'] = 0.0

        # 3. 检索覆盖率（检索文档与问题的相关性）
        if retrieved_docs:
            metrics['coverage'] = self._compute_coverage(query, retrieved_docs)
        else:
            metrics['coverage'] = 0.0

        # 4. MRR（平均倒数排名）
        if distances:
            # 假设距离越小，排名越高
            ranked = sorted(enumerate(distances), key=lambda x: x[1])
            mrr = sum([1.0 / (rank + 1) for rank, dist in ranked if dist < 0.5]) / len(ranked)
            metrics['mrr'] = mrr
        else:
            metrics['mrr'] = 0.0

        return metrics

    def _compute_relevance(self, query: str, answer: str) -> float:
        """
        计算答案与问题的相关性

        Args:
            query: 问题
            answer: 答案

        Returns:
            相关性得分（0-1）
        """
        try:
            # 使用嵌入模型计算相似度
            query_embedding = self.embedding_model.encode(query)
            answer_embedding = self.embedding_model.encode(answer)

            # 计算余弦相似度
            similarity = np.dot(query_embedding, answer_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(answer_embedding)
            )

            return float(similarity)
        except Exception as e:
            print(f"Error computing relevance: {e}")
            return 0.0

    def _compute_completeness(self, answer: str, context: str) -> float:
        """
        计算答案完整性（答案对上下文的覆盖程度）

        Args:
            answer: 答案
            context: 上下文

        Returns:
            完整性得分（0-1）
        """
        try:
            # 提取答案中的关键词
            answer_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', answer))

            # 提取上下文中的关键词
            context_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', context))

            if not answer_keywords:
                return 0.0

            # 计算答案关键词在上下文中的比例
            overlap = len(answer_keywords & context_keywords)
            completeness = overlap / len(answer_keywords)

            return float(completeness)
        except Exception as e:
            print(f"Error computing completeness: {e}")
            return 0.0

    def _compute_fluency(self, answer: str) -> float:
        """
        计算答案流畅性（基于句子结构）

        Args:
            answer: 答案

        Returns:
            流畅性得分（0-1）
        """
        try:
            # 分割句子
            sentences = re.split(r'[。！？\n]', answer)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return 0.0

            # 计算平均句子长度
            avg_sentence_length = np.mean([len(s) for s in sentences])

            # 计算句子长度方差（流畅性：方差越小越好）
            length_variance = np.var([len(s) for s in sentences])

            # 归一化得分（假设理想句子长度为20-50字，方差越小越好）
            if 20 <= avg_sentence_length <= 50:
                length_score = 1.0
            else:
                length_score = max(0, 1 - abs(avg_sentence_length - 35) / 35)

            variance_score = max(0, 1 - length_variance / 100)

            fluency = (length_score + variance_score) / 2

            return float(fluency)
        except Exception as e:
            print(f"Error computing fluency: {e}")
            return 0.0

    def _compute_information_density(self, answer: str) -> float:
        """
        计算答案信息密度（信息量与长度的比率）

        Args:
            answer: 答案

        Returns:
            信息密度得分（0-1）
        """
        try:
            # 提取关键词（2字以上的中文词）
            keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', answer)

            if not keywords or len(answer) == 0:
                return 0.0

            # 计算原始密度（关键词数 / 总字符数）
            raw_density = len(keywords) / len(answer)

            # 使用经验阈值进行归一化
            # 根据中文文本统计，正常的信息密度范围大约在0.05-0.15之间
            # 低于0.05说明信息稀疏，高于0.15可能包含重复或冗余信息
            min_threshold = 0.05  # 最小合理密度
            max_threshold = 0.15  # 最大合理密度

            # 使用分段线性归一化
            if raw_density < min_threshold:
                # 低于最小阈值：线性映射到0-0.3
                normalized = (raw_density / min_threshold) * 0.3
            elif raw_density > max_threshold:
                # 高于最大阈值：线性映射到0.7-1.0
                excess = raw_density - max_threshold
                normalized = 0.7 + min(excess / max_threshold, 0.3)
            else:
                # 在合理范围内：线性映射到0.3-0.7
                normalized = 0.3 + ((raw_density - min_threshold) / (max_threshold - min_threshold)) * 0.4

            # 确保在0-1范围内
            normalized = max(0.0, min(1.0, normalized))

            return float(normalized)
        except Exception as e:
            print(f"Error computing information density: {e}")
            return 0.0

    def _compute_diversity(self, docs: List[str]) -> float:
        """
        计算检索文档的多样性

        Args:
            docs: 文档列表

        Returns:
            多样性得分（0-1）
        """
        try:
            if len(docs) < 2:
                return 0.0

            # 计算每对文档的相似度
            similarities = []
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    emb1 = self.embedding_model.encode(docs[i])
                    emb2 = self.embedding_model.encode(docs[j])

                    sim = np.dot(emb1, emb2) / (
                        np.linalg.norm(emb1) * np.linalg.norm(emb2)
                    )
                    similarities.append(sim)

            # 多样性 = 1 - 平均相似度
            diversity = 1 - np.mean(similarities)

            return float(diversity)
        except Exception as e:
            print(f"Error computing diversity: {e}")
            return 0.0

    def _compute_coverage(self, query: str, docs: List[str]) -> float:
        """
        计算检索覆盖率（检索文档对问题的覆盖程度）

        Args:
            query: 问题
            docs: 文档列表

        Returns:
            覆盖率得分（0-1）
        """
        try:
            # 提取问题中的关键词
            query_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', query))

            if not query_keywords:
                return 0.0

            # 检查关键词是否在文档中出现
            covered_keywords = set()
            for doc in docs:
                doc_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', doc))
                covered_keywords.update(query_keywords & doc_keywords)

            # 计算覆盖率
            coverage = len(covered_keywords) / len(query_keywords)

            return float(coverage)
        except Exception as e:
            print(f"Error computing coverage: {e}")
            return 0.0

    def aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        聚合多个样本的指标

        Args:
            metrics_list: 指标列表

        Returns:
            聚合指标字典（均值和标准差）
        """
        if not metrics_list:
            return {}

        aggregated = {}

        # 获取所有指标名称
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())

        # 计算均值和标准差
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in metrics_list]
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        return aggregated