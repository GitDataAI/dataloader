import jiaozifs_client
from torch.utils.data import Dataset
from jiaozifs_client import V0Signer
from jiaozifs_client.models import FullTreeEntry
from typing import Callable, Optional
from typing import Callable, List, Optional


class JiaozifsDataset(Dataset):
    def __init__(
        self,
        owner,
        repo,
        ak,
        sk,
        url="https://api.jiaozifs.com/api/v1",
        refName="main",
        type="branch",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.owner = owner
        self.repo = repo
        self.type = type
        self.refName = refName

        configuration = jiaozifs_client.Configuration()
        configuration.host = url
        configuration.signer = V0Signer(ak, sk)
        self.apiClient = jiaozifs_client.ApiClient(configuration=configuration)

    def load_files(self, pattern: str):
        return jiaozifs_client.ObjectsApi(self.apiClient).get_files(
            self.owner, self.repo, self.refName, self.type, pattern=pattern
        )

    def load_sub_dirs(self, path: str):
        entries: List[FullTreeEntry] = jiaozifs_client.CommitApi(
            self.apiClient
        ).get_entries_in_ref(
            self.owner, self.repo, self.type, path=path, ref=self.refName
        )
        return [entry.name for entry in entries if entry.is_dir]

    def load_sub_files(self, path: str):
        entries: List[FullTreeEntry] = jiaozifs_client.CommitApi(
            self.apiClient
        ).get_entries_in_ref(
            self.owner, self.repo, self.type, path=path, ref=self.refName
        )
        return [entry.name for entry in entries if not entry.is_dir]

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
