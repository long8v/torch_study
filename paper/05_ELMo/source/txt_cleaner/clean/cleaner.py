import gc
import time
import pandas as pd

from konlpy.tag import Mecab

from .master import MasterCleaner


def epoch_time(prev_time):
    now_time = time.time()
    elapsed_time = now_time - prev_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Cleaner:
    def __init__(self, file_path, target_column, tag_column, config):
        self.cleaner = MasterCleaner(config, debug=False)
        self.df = None
        self.file_path = file_path
        self.target_column = target_column
        self.tag_column = tag_column
        self.mecab = Mecab()
        self.start_time = time.time()
        self.idx = 0

    def load_df(self, now_file_path, drop_duplicated_row="str_title"):
        del self.df
        gc.collect()
        df = pd.read_json(now_file_path).T
        df.drop_duplicates(drop_duplicated_row, inplace=True, keep="first")
        df.reset_index(inplace=True)
        df["str_all"] = df["str_title"] + " " + df["str_content"]
        print("size {} df {} is loaded".format(df.shape, now_file_path))
        minute, seconds = epoch_time(self.start_time)
        print(f"Loading: {self.idx+1:02} | Time: {minute}m {seconds}s")
        self.idx += 1
        self.df = df

    def __iter__(self):
        for file_name in self.file_path:
            self.load_df(file_name)
            for row_idx, row in self.df.iterrows():
                finished = False
                if row_idx == self.df.shape[0] - 1:
                    finished = True
                text = row[self.target_column]
                tag = row[self.tag_column]
                cleaned = self.cleaner.cleaning(text)
                words = self.mecab.morphs(cleaned)
                yield tag, " ".join(words), finished
