import json
import logging
import os


class Wikipedia:
    """Wikipedia Dataset

    Expecting this dataset to be represented as a collection of files that each contains multiple entries,
      organized one per line as a json object
    """
    def __init__(self, prefix=None):
        self.prefix = prefix
        self.files = [f for f in os.listdir(prefix) if f.startswith("wiki")]
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'Wikipedia Dataset initialized successfully. Total Files: {len(self.files)}.')

    def __iter__(self):
        for f in self.files:
            self.logger.info(f'Reading from {f}...')
            with open(f'{self.prefix}/{f}', "r") as fin:
                for line in fin:
                    if line:
                        yield json.loads(line)
