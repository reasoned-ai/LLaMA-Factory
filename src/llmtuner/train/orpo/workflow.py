# Inspired by: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

import os
import json
import torch
from typing import TYPE_CHECKING, List, Optional

from ...data import PairwiseDataCollatorWithPadding, get_dataset, split_dataset
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...extras.logging import get_logger
from ...hparams import ModelArguments
from ...model import load_model, load_tokenizer
from ..utils import create_modelcard_and_push
from .trainer import CustomORPOTrainer
from .metric import compute_metrics

logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments


def run_orpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = PairwiseDataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Update arguments
    training_args.remove_unused_columns = False  # important for pairwise dataset

    # Initialize our Trainer
    trainer = CustomORPOTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        **split_dataset(dataset, data_args, training_args),
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "rewards/accuracies", "sft_loss"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = []
        for d in dataset:
            label = tokenizer.decode(d["chosen_ids"]).replace(tokenizer.eos_token, "")
            input_ids = torch.tensor(d["prompt_ids"]).unsqueeze(0).to(model.device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=generating_args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=generating_args.do_sample,
                temperature=generating_args.temperature,
                top_p=generating_args.top_p,
            )
            pred = tokenizer.decode(
                outputs[0][input_ids.shape[-1] :], skip_special_tokens=True
            )
            predict_results.append({"label": label, "predict": pred})

        output_prediction_file = os.path.join(
            training_args.output_dir, "generated_predictions.jsonl"
        )
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res = [json.dumps(r, ensure_ascii=False) for r in predict_results]
            writer.write("\n".join(res))
        logger.info(f"Saving prediction results to {output_prediction_file}")

        prediction_metrics = compute_metrics(predict_results=predict_results)
        output_predict_results = os.path.join(
            training_args.output_dir, "predict_results.json"
        )
        with open(output_predict_results, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(prediction_metrics, ensure_ascii=False))
        logger.info(f"Saving prediction_metrics to {output_predict_results}")

    # Create model card
    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args
    )
