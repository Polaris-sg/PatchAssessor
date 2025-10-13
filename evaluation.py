import argparse
import numpy as np
import os
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from utils.utils import load_jsonl, make_needed_dir


class PatchCorrectnesEvaluator:
    """
    补丁正确性评估器，用于计算补丁正确性预测的各种评估指标
    """
    def __init__(self, predictions_path=None, predictions=None):
        """
        初始化评估器
        
        Args:
            predictions_path (str): 预测结果的JSON文件路径
            predictions (list): 预测结果对象列表
        """
        if predictions is not None:
            self.predictions = predictions
        elif predictions_path is not None:
            self.predictions = load_jsonl(predictions_path)
        else:
            self.predictions = []
            
        # 提取标签和预测值
        self.true_labels = []
        self.pred_labels = []
        
        for pred in self.predictions:
            true_label = pred.get('true_label', -1)
            pred_label = pred.get('prediction', -1)
            
            # 只考虑有效预测结果
            if true_label != -1 and pred_label != -1:
                self.true_labels.append(true_label)
                self.pred_labels.append(pred_label)
                
        self.true_labels = np.array(self.true_labels)
        self.pred_labels = np.array(self.pred_labels)
    
    def compute_metrics(self):
        """
        计算评估指标
        
        Returns:
            dict: 包含各项评估指标的字典
        """
        if len(self.true_labels) == 0:
            return {
                "error": "No valid predictions found"
            }
            
        # 计算基本指标
        metrics = {
            "total_samples": len(self.true_labels),
            "correct_samples": sum(self.true_labels == self.pred_labels),
            "accuracy": accuracy_score(self.true_labels, self.pred_labels)
        }
        
        # 计算F1分数
        metrics["f1_score"] = f1_score(self.true_labels, self.pred_labels)
        
        # 计算AUC (如果有正例和负例)
        positive_samples = sum(self.true_labels == 1)
        negative_samples = sum(self.true_labels == 0)
        
        if positive_samples > 0 and negative_samples > 0:
            metrics["auc"] = roc_auc_score(self.true_labels, self.pred_labels)
        else:
            metrics["auc"] = None
            
        # 计算混淆矩阵
        cm = confusion_matrix(self.true_labels, self.pred_labels)
        metrics["confusion_matrix"] = cm.tolist()
        
        # 计算精确率和召回率
        if len(cm) > 1:
            tn, fp, fn, tp = cm.ravel()
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            metrics["precision"] = None
            metrics["recall"] = None
            
        return metrics
    
    def print_metrics(self, metrics=None):
        """
        打印评估指标
        
        Args:
            metrics (dict): 评估指标字典，如果为None则会自动计算
        """
        if metrics is None:
            metrics = self.compute_metrics()
            
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            return
            
        print("\n=== Patch Correctness Evaluation Metrics ===")
        print(f"Total samples: {metrics.get('total_samples', 0)}")
        print(f"Correct predictions: {metrics.get('correct_samples', 0)}")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
        
        if metrics.get("auc") is not None:
            print(f"AUC: {metrics.get('auc', 0):.4f}")
        else:
            print("AUC: Not available (need both positive and negative samples)")
            
        print("\nConfusion Matrix:")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        
        if len(cm) > 1:
            print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
            print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
            print("(0=正确补丁, 1=不正确补丁)")
            
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            
            if precision is not None:
                print(f"Precision: {precision:.4f}")
            if recall is not None:
                print(f"Recall: {recall:.4f}")
        else:
            print("Confusion matrix not available")
    
    def save_metrics(self, output_path):
        """
        保存评估指标到文件
        
        Args:
            output_path (str): 输出文件路径
        """
        metrics = self.compute_metrics()
        make_needed_dir(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Metrics saved to {output_path}")


def main(args):
    # 创建评估器
    evaluator = PatchCorrectnesEvaluator(args.predictions_path)
    
    # 计算指标
    metrics = evaluator.compute_metrics()
    
    # 打印指标
    evaluator.print_metrics(metrics)
    
    # 保存指标
    if args.output_path:
        evaluator.save_metrics(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate patch correctness predictions')
    parser.add_argument('--predictions_path', required=True, help='Path to the prediction results JSONL file')
    parser.add_argument('--output_path', help='Path to save evaluation metrics in JSON format')
    
    args = parser.parse_args()
    main(args)