# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BrWac clean dataset"""

_CITATION = """
@inproceedings{wagner2018brwac,
  title={The brwac corpus: A new open resource for brazilian portuguese},
  author={Wagner Filho, Jorge A and Wilkens, Rodrigo and Idiart, Marco and Villavicencio, Aline},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
"""

_DESCRIPTION = """
The BrWaC (Brazilian Portuguese Web as Corpus) is a large corpus constructed following the Wacky framework,
which was made public for research purposes. The current corpus version, released in January 2017, is composed by
3.53 million documents, 2.68 billion tokens and 5.79 million types. Please note that this resource is available
solely for academic research purposes, and you agreed not to use it for any commercial applications.
Manually download at https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC
"""

_HOMEPAGE = "https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC"

_LICENSE = ""


import collections
import gzip

import datasets


logger = datasets.logging.get_logger(__name__)

_BASE_DIR = "/home/diogo/extracted/brwac-clean/"
_BASE_DATA_URL = "/home/diogo/extracted/brwac-clean/data/"


class BrwacCleanConfig(datasets.BuilderConfig):
    """BRWAC-clean corpus."""

    def __init__(self, **kwargs):
        # Initialize the base class.
        name = "brwac-clean"
        description = "brwac-clean dataset"
        super(BrwacCleanConfig, self).__init__(name=name, description=description, **kwargs)

        # Additional attributes
        self.base_data_url = _BASE_DATA_URL


class BrwacClean(datasets.GeneratorBasedBuilder):
    """BRWAC corpus."""

    BUILDER_CONFIGS = [
        BrwacCleanConfig(
            version=datasets.Version("1.0.0"),
        )
    ]
    BUILDER_CONFIG_CLASS = BrwacCleanConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"id": datasets.Value("int64"), "text": datasets.Value("string")}),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        checksum_url = _BASE_DIR + "file_names.txt"
        checksum_file = dl_manager.download(checksum_url)

        with open(checksum_file, encoding="utf-8") as f:
            data_filenames = [line.split("\t")[0] for line in f if line]
            data_urls = [self.config.base_data_url + data_filename.strip() for data_filename in data_filenames]
        downloaded_files = dl_manager.download(data_urls)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": downloaded_files}),
        ]

    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    feature = id_, {"id": id_, "text": line.replace("<END>", "\n").rstrip()}
                    yield feature
                    id_ += 1
