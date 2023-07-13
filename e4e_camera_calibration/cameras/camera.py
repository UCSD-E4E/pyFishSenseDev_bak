from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import List

import numpy as np


class Camera(ABC):
    def __init__(
        self,
        name: str = None,
        manufacturer: str = None,
        model: str = None,
        serial_number: str = None,
    ) -> None:
        super().__init__()

        self._image_files: List[str] = None
        self._image_files_idx = 0
        self._name = name
        self._manufacturer = manufacturer
        self._model = model
        self._serial_number = serial_number
        self._image_count = 0
        self._load_from_directory = False
        self._load_from_video = False

    @property
    def image_count(self):
        return self._image_count

    @property
    def manufacturer(self):
        return self._manufacturer

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name

    @abstractproperty
    def number_of_sensors(self) -> int:
        raise NotImplementedError()

    @property
    def serial_number(self):
        return self._serial_number

    def at(self, index: int):
        self.seek(index)
        return self.__next__()

    def load(self, file_path: str):
        path = Path(file_path)

        self._load_from_directory = False
        self._load_from_video = False

        if self._is_image_file(path):
            self._image_files = self._process_directory(file_path)
            self._image_count = len(self._image_files)
            self._image_files_idx = 0
            self._load_from_directory = True
        elif self._is_video_file(path):
            self._load_video(file_path)
            self._load_from_video = True
        else:
            raise NotImplementedError()

    def seek(self, index: int):
        if self._load_from_directory:
            self._seek_directory(index)
        elif self._load_from_video:
            self._seek_video_file(index)
        else:
            raise NotImplementedError()

    def _is_image_file(self, path: Path):
        return path.is_dir()

    def _is_video_file(self, path: Path):
        return path.suffix == ".mp4"

    @abstractmethod
    def _load_image_file(self, file_path: str):
        raise NotImplementedError()

    @abstractmethod
    def _load_video(self, file_path: str):
        raise NotImplementedError()

    def _next_from_directory(self):
        if self._image_files_idx < len(self._image_files):
            file = self._image_files[self._image_files_idx]
            self._image_files_idx += 1

            image = self._load_image_file(file)

            return self._preprocess_image(image)

        raise StopIteration

    def _next_from_video_file(self):
        image = self._next_video_frame()

        if image is None:
            raise StopIteration

        return self._preprocess_image(image)

    @abstractmethod
    def _next_video_frame(self):
        raise NotImplementedError()

    @abstractmethod
    def _preprocess_image(self, image: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _process_directory(self, file_path: str):
        raise NotImplementedError()

    def _seek_directory(self, index: int):
        self._image_files_idx = index

    @abstractmethod
    def _seek_video_file(self, index: int):
        raise NotImplementedError()

    @abstractmethod
    def _write_image_file(self, image: np.ndarray, file_path: Path):
        raise NotImplementedError()

    # Used in calibrated_stereo_camera.py to get the left and right images
    def __iter__(self):
        return self

    def __next__(self):
        if self._load_from_directory:
            return self._next_from_directory()
        elif self._load_from_video:
            return self._next_from_video_file()

        raise NotImplementedError()
