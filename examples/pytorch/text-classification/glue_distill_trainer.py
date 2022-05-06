import numpy
import torch
import torch.nn.functional as F

from transformers import AutoModelForQuestionAnswering, Trainer
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from collections import defaultdict
import math
import os

import numpy

class GlueDistillTrainer(Trainer):
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
            raise ValueError("DistillTrainer requires a valid teacher")

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

            logits_label = inputs['labels']
            logits_student = outputs["logits"]
            label_loss = self.criterion(logits_student, logits_label)

            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            logits_teacher = teacher_output["logits"]

            teacher_loss = (
                F.kl_div(
                    input=F.log_softmax(logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )

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

