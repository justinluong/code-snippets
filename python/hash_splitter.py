from functools import cached_property, singledispatch
from typing import Union
from zlib import crc32

import numpy as np
import pandas as pd


class HashSplitter:
    def __init__(
        self,
        df: pd.DataFrame,
        id_column: str,
        test_ratio: float,
        validation_ratio: float = 0,
    ):
        assert 0 < test_ratio < 1, "invalid test_ratio"
        assert 0 <= validation_ratio < 1, "invalid validation_ratio"
        assert (
            test_ratio + validation_ratio < 1
        ), "invalid test_ratio and validation_ratio combination"

        self.__og_df = df
        self.__id_column = id_column
        self.__test_ratio = test_ratio
        self.__validation_ratio = validation_ratio

    def check_split(self, id: Union(int, str)) -> str:
        byteslike_identifier = self.get_byteslike_identifier(id)

        # Old versions of crc32 outputs a signed value, remove sign
        hashed_id = crc32(byteslike_identifier) & 0xFFFFFFFF

        max_hash = 2**32 - 1
        test_boundary = max_hash - max_hash * self.__test_ratio
        validation_boundary = test_boundary - max_hash * self.__validation_ratio

        if hashed_id >= test_boundary:
            return "test"
        elif hashed_id < test_boundary and hashed_id >= validation_boundary:
            return "validation"
        else:
            return "training"

    @staticmethod
    @singledispatch
    def get_byteslike_identifier(id: Union(int, str)):
        raise NotImplementedError("Byteslike conversion not implemented for this type")

    @get_byteslike_identifier.register
    def _(self, id: int):
        return np.int64(id)

    @get_byteslike_identifier.register
    def _(self, id: str):
        return bytes(str(id), "utf-8")

    @property
    def dataset(self):
        return self.__og_df

    @cached_property
    def dataset_with_split(self) -> pd.DataFrame:
        split_assignment = self.__og_df[self.__id_column].apply(
            lambda x: self.check_split(x)
        )
        return self.__og_df.assign(split_assignment=split_assignment)

    @cached_property
    def training_set(self) -> pd.DataFrame:
        return self.dataset_with_split[
            self.dataset_with_split["split_assignment"] == "training"
        ].drop(axis=1, columns="split_assignment")

    @cached_property
    def test_set(self) -> pd.DataFrame:
        return self.dataset_with_split[
            self.dataset_with_split["split_assignment"] == "test"
        ].drop(axis=1, columns="split_assignment")

    @cached_property
    def validation_set(self) -> pd.DataFrame:
        return self.dataset_with_split[
            self.dataset_with_split["split_assignment"] == "validation"
        ].drop(axis=1, columns="split_assignment")

    @cached_property
    def split_data(self):
        if self.__validation_ratio == 0:
            return self.training_set, self.test_set
        else:
            return self.training_set, self.test_set, self.validation_set
