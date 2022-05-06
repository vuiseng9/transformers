import numpy
import torch
import torch.nn.functional as F

from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoModelForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from collections import defaultdict
import math
import os

import numpy

class QADistillTrainer(QuestionAnsweringTrainer):
    """
    Question Answering trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """
    def __init__(self, *args, teacher=None, hardness=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher is None:
            raise ValueError("QADistillTrainer requires a valid teacher")

        self.teacher = teacher.to(kwargs['args'].device)
        self.distill_hardness = hardness
        self.distill_temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_counter = 0
        self.metrics = defaultdict(float)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        self.loss_counter += 1
        outputs = model(**inputs)
        if self.teacher is None:
            loss = outputs["loss"]
        else:
            start_logits_student = outputs["start_logits"]
            end_logits_student = outputs["end_logits"]
            start_logits_label = inputs["start_positions"]
            end_logits_label = inputs["end_positions"]
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            start_logits_teacher = teacher_output["start_logits"]
            end_logits_teacher = teacher_output["end_logits"]
            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
            loss_start = self.criterion(start_logits_student, start_logits_label)
            loss_end = self.criterion(end_logits_student, end_logits_label)
            label_loss = (loss_start + loss_end) / 2.0
            self.metrics["label_loss"] += float(label_loss)
            self.metrics["teacher_loss"] += float(teacher_loss)
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)

        return (loss, outputs) if return_outputs else loss

    def log(self, logs):
        if self.loss_counter != 0:
            for k, v in self.metrics.items():
                logs[k] = float(v) / self.loss_counter

            self.loss_counter = 0
            self.metrics = defaultdict(float)

        return super().log(logs)

