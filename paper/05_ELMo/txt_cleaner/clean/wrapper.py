import os

from .cleaner import Cleaner


class CleanerWrapper:
    def __init__(self, dir_path, config, year=None, month=None):
        self.dir_path = dir_path
        self.year = year
        self.month = month
        self.file_list = sorted(self.get_file_list())
        self.corpus = Cleaner(self.file_list, 'str_all', '_id', config)
        self.df_idx = 0

    def get_file_list(self):
        file_list = list()
        for file in os.listdir(self.dir_path):
            if self.year is None:
                if not file.endswith('json'):
                    continue
                file_list.append(os.path.join(self.dir_path, file))
            elif self.month is None:
                if '{}'.format(self.year) in file:
                    file_list.append(os.path.join(self.dir_path, file))
            elif self.month is not None:
                if '{}_{:02}'.format(self.year, self.month) in file:
                    file_list.append(os.path.join(self.dir_path, file))
        return file_list

    def __iter__(self):
        return self.corpus.__iter__()
