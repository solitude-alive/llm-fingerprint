from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
import torch
import random
from pathlib import Path
from datasets import load_dataset

from .base_model import BaseModel
from utils import logger
from utils.format import get_formatter
from utils.utils import match_and_extract
from utils.logits_utils import process_logits


class GemmaModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.huggingface = args["huggingface"]
        self.model = None  # Standard model
        self.tokenizer = None

        self.weight = None
        self.logits = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================== Load Model and Weight ==================
    def load_model(self):
        assert self.huggingface is True, "GemmaModel only supports huggingface models"
        model_path = os.path.join(self.args["standard"], self.args["version"])
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def load_lora_model(self, path):
        assert self.huggingface is True, "GemmaModel only supports huggingface models"
        self.load_model()  # Reset the model
        model_lora = PeftModel.from_pretrained(self.model, path)

        del self.model
        torch.cuda.empty_cache()

        return model_lora

    def load_qlora_model(self, path):
        assert self.huggingface is True, "GemmaModel only supports huggingface models"
        self.load_model()  # Reset the model
        model_qlora = PeftModel.from_pretrained(self.model, path, load_in_4bit=True)

        del self.model
        torch.cuda.empty_cache()

        return model_qlora

    def load_matrix(self):
        last_layer = self.model.get_output_embeddings()
        assert isinstance(last_layer, torch.nn.Linear), \
            f"The last layer of the model is not a linear layer, but got {type(last_layer)}."

        self.weight = last_layer.weight

        name = self.args["name"]
        logger.log(f"Load weight matrix from model {name} successfully.")

    # ================== Model Generate ==================
    @torch.no_grad()
    def _model_generate(self, model, *, text=None, dataset=None, fine_tuned=False):

        model = model.to(self.device)

        outputs_text = []
        logits = []
        min_len = 300
        max_new_tokens = 10

        if text is None and dataset is None:
            text = [
                "This is Lecun speaking, I am a professor at NYU and the chief AI scientist at Facebook,",
                "Once upon a time,",
                "My name is Julien and I like to",
                "The quick brown fox jumps over the lazy dog,",
                "I am a student at the University of Toronto,",
                "We are living in a simulation,",
                "In the beginning God created the heavens and the earth,",
                "The year is 2024,",
                "The World War III has begun in,",
                "The COVID-19 pandemic has caused a lot of damage to the world,",
                "The stock market is crashing,",
                "Education is the most powerful weapon which you can use to change the world,",
                "Abraham Lincoln was the 16th president of the United States,",
                "Apollo 11 was the first manned mission to land on the moon,",
                "The United Union is a political and economic union of 27 member states located primarily in Europe,",
                "Artificial intelligence is intelligence demonstrated by machines,",
                "The Turing test is a test of a machine's ability to exhibit intelligent behavior equivalent to,"
                "What amazing weather today,",
                "How wonderful it is to be alive,",
                "LeBron James is a professional basketball player,",    # Stop here for fine-tuned model
                "Stephen Curry won the NBA MVP award in 2015,",
                "Lin Dan is a retired Chinese badminton player,",
                "Aristotle was an ancient Greek philosopher and scientist,",
                "Albert Einstein was a German-born theoretical physicist,",
                "Nuclear weapon is a weapon of mass destruction,",
                "The United Nations is an intergovernmental organization,"
                "Pandas are bears native to south-central China,",
                "The Amazon rainforest is the world's largest tropical rainforest,",
                "The Great Wall of China is a series of fortifications made of stone,"
            ]
            max_new_tokens = 100
        elif dataset is not None:
            text = []
            data = load_dataset(dataset, split="train")
            formatter = get_formatter(dataset.split("/")[-1])
            nums = random.sample(range(len(data)), 30)
            max_new_tokens = 15
            for i in nums:
                text.append(formatter.format_prompt((data[i])))

        for input_text in text:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     output_logits=True,
                                     return_dict_in_generate=True)
            output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            output_logits = outputs.logits

            logger.log(output_text)
            outputs_text.append(output_text)
            logits.extend(output_logits)

            if len(logits) > min_len:
                break

        logger.log(f"Get {len(logits)} logits from the model.")

        logits = tuple(logits)

        return outputs_text, logits

    # ================== Process with oracle model and lora model ==================
    def _process_oracle(self):
        self.load_model()
        self.load_matrix()

        w = self.weight

        logger.log("=====Start Testing Oracle Model=====")

        text, logits = self._model_generate(self.model)
        residuals, rank = process_logits(w, logits, type_logits=["all"])

        logger.log(f"==Rank: {rank}==")
        logger.log(f"==Average residual: {residuals}==")
        logger.log("=====Finish Testing Oracle Model=====")

    def _process_lora(self):
        if self.weight is None:
            self.load_model()
            self.model.to(self.device)
            self.load_matrix()

        w = self.weight

        model_dir = self.args["path"]["dir"]
        dataset = self.args["dataset"]

        for data_name in dataset:

            logger.log(f"=======Dataset: {data_name}=======")

            for path in os.listdir(str(os.path.join(model_dir, data_name))):

                match = match_and_extract(path)  # Match the model name
                if match is None:
                    continue
                data, method, suffix = match
                logger.log(f"{data}-{method}-{suffix}")

                sub_path = os.path.join(model_dir, data_name, path, self.args["version"], self.args["path"]["sub"])
                sub_path_dir = Path(str(sub_path))

                for model_path in sub_path_dir.glob("*"):
                    if model_path.is_dir():
                        logger.log(f"==Start Testing Model{str(model_path)}==")

                        data_path = os.path.join(os.path.dirname(os.path.dirname(model_dir)), "datasets", data_name)
                        model_lora = self.load_lora_model(model_path)
                        text, logits = self._model_generate(model_lora, dataset=data_path)

                        # delete the model to save GPU memory
                        del model_lora
                        torch.cuda.empty_cache()

                        residuals, rank = process_logits(w, logits, type_logits=["all"])

                        logger.log(f"==Rank: {rank}==")
                        logger.log(f"==Average residual: {residuals}==")
                        logger.log("=====Finish Testing LORA Model=====")

    def _process_qlora(self):
        if self.weight is None:
            self.load_model()
            self.model.to(self.device)
            self.load_matrix()

        w = self.weight

        model_dir = self.args["path"]["dir"]
        dataset = self.args["dataset"]

        for data_name in dataset:

            logger.log(f"=======Dataset: {data_name}=======")

            for path in os.listdir(str(os.path.join(model_dir, data_name))):

                match = match_and_extract(path)  # Match the model name
                if match is None:
                    continue
                if match[1] != "qlora":
                    continue
                data, method, suffix = match
                logger.log(f"{data}-{method}-{suffix}")

                sub_path = os.path.join(model_dir, data_name, path, self.args["version"], self.args["path"]["sub"])
                sub_path_dir = Path(str(sub_path))

                for model_path in sub_path_dir.glob("*"):
                    if model_path.is_dir():
                        logger.log(f"==Start Testing Model{str(model_path)}==")

                        data_path = os.path.join(os.path.dirname(os.path.dirname(model_dir)), "datasets", data_name)
                        model_qlora = self.load_qlora_model(model_path)
                        text, logits = self._model_generate(model_qlora, dataset=data_path)

                        # delete the model to save GPU memory
                        del model_qlora
                        torch.cuda.empty_cache()

                        residuals, rank = process_logits(w, logits, type_logits=["all"])

                        logger.log(f"==Rank: {rank}==")
                        logger.log(f"==Average residual: {residuals}==")
                        logger.log("=====Finish Testing QLORA Model=====")

    def _process_version(self):
        if self.weight is None:
            self.load_model()
            self.model.to(self.device)
            self.load_matrix()

        w = self.weight

        logger.log("=====Start Testing New Version Model=====")

        model_path = os.path.join(self.args["standard"], self.args["new_version"])
        model_version = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer_version = AutoTokenizer.from_pretrained(model_path)

        text, logits = self._model_generate(model_version)

        # delete the model to save GPU memory
        del model_version
        torch.cuda.empty_cache()

        residuals, rank = process_logits(w, logits, type_logits=["all"])
        logger.log(f"==Rank: {rank}==")
        logger.log(f"==Average residual: {residuals}==")
        logger.log("=====Finish Testing Testing New Version Model=====")

    # ================== Call ==================
    def __call__(self, *args, **kwargs):
        self._process_oracle()
        self._process_lora()
        self._process_qlora()
        self._process_version()
