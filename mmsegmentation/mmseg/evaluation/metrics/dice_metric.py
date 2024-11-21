from collections import OrderedDict
import numpy as np
from prettytable import PrettyTable
import torch
from mmseg.registry import  METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmsegmentation.configs._base_.datasets.XRayDataset import  CLASSES

@METRICS.register_module()
class DiceMetric(BaseMetric):
    def __init__(self,
                 collect_device='cpu',
                 prefix=None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = y_true.flatten(-2)
        y_pred_f = y_pred.flatten(-2)

        intersection = torch.sum(y_true_f * y_pred_f, -1)

        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def process(self, data_batch, data_samples):
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        
            
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data']
            label = data_sample['gt_sem_seg']['data'].to(pred_label)   
            self.results.append(
                self.dice_coef(label, pred_label)
            )

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        results = torch.stack(self.results, 0)
        dices_per_class = torch.mean(results, 0)
        avg_dice = torch.mean(dices_per_class)

        ret_metrics = {
            "Dice": dices_per_class.detach().cpu().numpy(),
        }
        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        metrics = {
            "mDice": torch.mean(dices_per_class).item()
        }

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': CLASSES})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics