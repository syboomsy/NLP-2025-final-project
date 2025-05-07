"""LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning"""

import re
import datasets

logger = datasets.logging.get_logger(__name__)


_HOMEPAGE = "https://github.com/lgw863/LogiQA-dataset"

_DESCRIPTION = """\
LogiQA is constructed from the logical comprehension problems from \
publically available questions of the National Civil Servants Examination \
of China, which are designed to test the civil servant candidates’ critical \
thinking and problem solving. This dataset includes the English versions only; \
the Chinese versions are available via the homepage/original source."""

_CITATION = """\
@article{liu2020logiqa,
  title={Logiqa: A challenge dataset for machine reading comprehension with logical reasoning},
  author={Liu, Jian and Cui, Leyang and Liu, Hanmeng and Huang, Dandan and Wang, Yile and Zhang, Yue},
  journal={arXiv preprint arXiv:2007.08124},
  year={2020}
}
"""

_URLS = {
    "en_train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "en_test": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
    "en_eval": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
}

def _process_answer(answer):
    if not any(answer.startswith(x) for x in "ABCD"):
        return answer
    else:
        return answer[3:]

def _process_sentences(text):
    text = text.replace("\n", "")
    sents = text.split(".")
    text = ""
    for sent in sents:
        if len(sent) == 0:
            continue
        if len(text) == 0:
            text += sent
        elif sent[0].isnumeric():
            text += "."+sent
        else:
            text += ". "+sent
    text = text.replace("  ", " ")
    text = text.replace("\\'", "'")
    while text.endswith(" "):
        text = text[:-1]
    if re.match('^[A-Z][\w\s]+[?.!]$', text) is None:
        text += "."
    text = text.replace("?.", "?")
    text = text.replace("!.", "!")
    text = text.replace("..", ".")
    return text

class LogiQA(datasets.GeneratorBasedBuilder):
    def _info(self):
        #  "question": question,
        #             "options" : options:dict,
        #             "answer": answer:stri,
        #             # "options": [_process_answer(answers[i]) for i in range(4)],
        #             "answer_idx": answer_idx
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer_idx": datasets.Value("string"),
                "options": datasets.Features({
                    "A": datasets.Value("string"),
                    "B": datasets.Value("string"),
                    "C": datasets.Value("string"),
                    "D": datasets.Value("string"),
                }),
                "answer": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["en_train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["en_eval"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["en_test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("generating examples from = %s", filepath)
        option_keys = "ABCD"
        with open(filepath, encoding="utf-8") as f:
            logiqa = f.readlines()
            logiqa = [_process_sentences(s) for s in logiqa]

            for key in range(int(len(logiqa)/8)):
                row = 8*key
                correct_answer = logiqa[row+1].replace(".","")
                context = logiqa[row+2]
                query = logiqa[row+3]
                answers = logiqa[row+4:row+8]



                question =  context + "\n\n" + query
                options = {
                    option_keys[i] :_process_answer(answers[i])
                      for i in range(len(option_keys))
                 }
                answer_idx = correct_answer.upper()
                answer = options[answer_idx]
                yield key, {
                    # "context": context,
                    "question": question,
                    "options" : options,
                    "answer": answer,
                    # "options": [_process_answer(answers[i]) for i in range(4)],
                    "answer_idx": answer_idx
                }

