from abc import ABC, abstractmethod


class Formatter(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def format_train(self, sample):
        pass

    @abstractmethod
    def format_prompt(self, sample):
        pass


class AlpacaFormatter(Formatter):
    def __init__(self):
        super().__init__("alpaca")

    def format_train(self, sample):
        message = f"""
            Below is an instruction that describes a task, paired with an input that provides further context. \
            "Write a response that appropriately completes the request.\n\n"\
            "### Instruction:\n{sample["instruction"]}\n\n### Input:\n{sample["input"]}\n\
            "### Response: {sample["output"]}\n"
            """
        return message

    def format_prompt(self, sample):
        message = f"""
            Below is an instruction that describes a task, paired with an input that provides further context. \
            "Write a response that appropriately completes the request.\n\n"\
            "### Instruction:\n{sample["instruction"]}\n\n### Input:\n{sample["input"]}\n\
            "### Response: "
            """
        return message


class SummarizeFormatter(Formatter):
    def __init__(self):
        super().__init__("samsum")

    def format_train(self, sample):
        message = f"""
            Summarize this dialogue:\n{sample["dialogue"]}\n---\nSummary:\n{sample["summary"]}\n
            """
        return message

    def format_prompt(self, sample):
        message = f"""
            Summarize this dialogue:\n{sample["dialogue"]}\n---\nSummary:\n
            """
        return message


# =========== API ==================

def get_formatter(name):
    if name == "tatsu-lab":
        return AlpacaFormatter()
    elif name == "samsum":
        return SummarizeFormatter()
    else:
        raise ValueError("Invalid formatter name, must be 'tatsu-lab' or 'samsum'")
