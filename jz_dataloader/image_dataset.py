from __future__ import absolute_import

import io
import os
from PIL import Image
from jz_dataloader.vision import VisionDataset
from typing import Callable, Dict, List, Optional, Tuple, Union

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp",
                  ".pgm", ".tif", ".tiff", ".webp")


def has_file_allowed_extension(
        filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(
        extensions if isinstance(
            extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(topDirs: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(topDirs)
    if not classes:
        raise FileNotFoundError(f"Couldn't find any top dir.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class ImageDataset(VisionDataset):
    def __init__(
        self,
        owner: str,
        repo: str,
        ak: str,
        sk: str,
        url="https://api.jiaozifs.com/api/v1",
        refName="main",
        type="branch",
        path="/",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(owner, repo, ak, sk, url, refName,
                         type, transforms, transform, target_transform)
        self.path = path

        classes, class_to_idx = self.find_classes(path)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.samples = self.make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample =  Image.open(io.BytesIO(self.load_object(path))).convert("RGB") 
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def find_classes(self, path: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        top_dirs = self.load_sub_dirs(path)
        last_names = [os.path.basename(path) for path in top_dirs]
        return find_classes(last_names)

    def make_dataset(
        self,
        base_path: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(base_path, target_class)
            # todo check dir exit
            files = self.load_files(os.path.join(target_dir, "*"))
            for path in sorted(files):
                if has_file_allowed_extension(path, extensions):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes

        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
