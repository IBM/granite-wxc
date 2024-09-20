import logging
import multiprocessing
import os
import queue
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
from filelock import FileLock, Timeout
from torch.utils.data import Dataset, IterableDataset

from granitewxc.utils.distributed import get_local_world_size, get_node_rank, get_world_size


def cache_file_name(cache_idx: int, cache_size: int) -> str:
    '''
    Args:
        cache_idx: Position within cache.
        cache_size: Size of cache (in objects).
    '''
    if (cache_idx < 0) or (cache_idx >= cache_size):
        raise ValueError('Ensure that 0 <= `cache_idx` < `cache_size`.')

    return f'cache_{str(cache_idx).zfill(len(str(cache_size)))}.pt'


class CachedDataset(IterableDataset):
    '''
    Iterable dataset to continuously sample samples from local cache as maintained by Cache object.

    See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for
    documentation on IterableDataset.
    '''

    def __init__(self, cache_dir: Path, cache_size: int, n_samples_per_epoch: int):
        '''
        Args:
            cache_dir: Location of cache.
            cache_size: Size of the cache.
            n_samples_per_epoch: Indicates how many samples should be returned per epoch.
        '''
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.n_samples_per_epoch = n_samples_per_epoch

    def __len__(self):
        return self.n_samples_per_epoch

    def __iter__(self):
        g = torch.Generator()
        index_order = torch.randperm(self.cache_size, generator=g).tolist()
        current_index = 0
        samples_returned = 0

        while True:
            cache_idx = index_order[current_index]
            file_name = self.cache_dir / cache_file_name(cache_idx, self.cache_size)

            lock_file_name = file_name.with_suffix('.lock')
            try:
                with FileLock(lock_file_name, timeout=0):
                    if os.path.isfile(file_name):
                        sample = torch.load(file_name)
                    else:
                        sample = None
            except Timeout:
                sample = None

            current_index += 1
            if current_index >= len(index_order):
                current_index = 0

            if sample is not None:
                yield sample
                samples_returned += 1

            if samples_returned >= self.n_samples_per_epoch:
                break


class Cache:
    '''
    Caches slow-to-retrieve dataset locally.

    Usage example:
    ```
    cache = Cache(
        dataset,
        'path/to/cache',
        cache_size=16,
        n_workers=2,
        clean_up=True
    )
    cache.start()

    data_loader = DataLoader(cache.cached_dataset, batch_size=16, num_workers=2)

    for epoch in range(epochs):
        cache.set_epoch(epoch)
        for idx, sample in enumerate(data_loader):
            # ...
    cache.shutdown()
    ```

    See the pytorch DistributedSampler code
    (https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler)
    regarding the sampling logic.
    '''

    def __init__(
        self,
        dataset: Dataset,
        cache_dir: Union[Path, str],
        cache_size: int = 64,
        n_workers: int = 1,
        num_replicas: Optional[int] = None,
        node_rank: Optional[int] = None,
        seed: int = 0,
        inactive: bool = False,
        clean_up: bool = True,
    ):
        '''
        For DDP and similar, only the process with LOCAL_RANK 0 should be active and
        all others inactive. Use `inactive` for this.

        Args:
            dataset: Dataset to be cached
            cache_dir: Local dataset to use for caching
            cache_size: Number os samples to store in cache
            n_workers: Number of processes to use to fill and refresh cache
            num_replicas: Number of nodes participating in
                distributed training. By default, this is
                `world_size` // `local_world_size` as retrieved from the
                current distributed group. Note that this assumes identical local world
                size on all nodes.
            node_rank: Rank of the current process group. By default, this is retrieved from the
                current distributed group.
            seed: random seed. Should be identical across all processes in the distributed
                group.
            inactive: Indicates that this particular instance should not load any data.
            clean_up: Delete all cached elements at shutdown.
        '''
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        if not cache_dir.is_dir():
            raise ValueError(f'{cache_dir} is not a directory.')

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                world_size = get_world_size()
                local_world_size = get_local_world_size()
                num_replicas = world_size // local_world_size
            else:
                num_replicas = 1
        if node_rank is None:
            if dist.is_available() and dist.is_initialized():
                node_rank = get_node_rank()
            else:
                node_rank = 0
        if node_rank >= num_replicas or node_rank < 0:
            raise ValueError(
                f"Invalid node_rank {node_rank}, node_rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.cache_dir = cache_dir
        self.cache_size = cache_size
        self.n_workers = n_workers
        self.num_replicas = num_replicas
        self.node_rank = node_rank
        self.seed = seed
        self.inactive = inactive
        self.clean_up = clean_up

        self.epoch = 0

        if self.inactive:
            self.queues = []
            self.log_queues = []
            self.workers = []
        else:
            self.queues = [multiprocessing.Queue(maxsize=len(dataset)) for i in range(n_workers)]
            self.log_queues = [multiprocessing.Queue(maxsize=1) for i in range(n_workers)]
            self.workers = []

        self.cached_dataset = CachedDataset(
            cache_dir=self.cache_dir, cache_size=cache_size, n_samples_per_epoch=len(dataset)
        )

    def set_epoch(self, epoch: int) -> None:
        '''
        Sets the epoch. This is relevant for sampling purposes.

        Args:
            epoch: Current epoch.
        '''

        self.epoch = epoch

        if not self.inactive:
            self.queue_indices()

    def queue_indices(self) -> None:
        '''
        Queue new indices for caching. The new set of indices will slowly replace previously cached data.
        '''

        if self.inactive:
            return

        if len(self.workers) == 0:
            raise RuntimeError('Start the workers before queueing indices.')

        for q in self.queues:
            q.put(-1)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = indices[self.node_rank : len(self.dataset) : self.num_replicas]

        for worker_idx in range(self.n_workers):
            for sample_idx in indices[worker_idx :: self.n_workers]:
                self.queues[worker_idx].put(sample_idx)

    def samples_retrieved(self) -> int:
        '''
        Obtain log data from workers.
        Returns:
            For active workers, number of samples retrieved since last call to `set_epoch` or `queue_indices`.
            For inactive workers, 0.
        '''

        if self.inactive:
            return 0

        if len(self.workers) == 0:
            raise RuntimeError('Start the workers before requesting logs indices.')

        # Ask all workers to submit logging information
        for q in self.queues:
            q.put(-2)

        samples_retrieved = [q.get() for q in self.log_queues]
        samples_retrieved = sum(samples_retrieved)

        return samples_retrieved

    def elements_cached(self) -> int:
        '''
        Returns number of elements currently cached.

        Returns:
            Number of cache slots currently occupied.
        '''
        n_cached = 0

        for cache_idx in range(self.cache_size):
            file_name = self.cache_dir / cache_file_name(cache_idx, self.cache_size)

            lock_file_name = file_name.with_suffix('.lock')
            try:
                with FileLock(lock_file_name, timeout=0):
                    if os.path.isfile(file_name):
                        n_cached += 1
            except Timeout:
                pass

        return n_cached

    def half_full(self) -> bool:
        '''
        Check if at least half the cache slots are occupied.

        Returns:
            True if at least half the cache slots are occupied. Otherwise false.
        '''

        return self.elements_cached() >= self.cache_size / 2.0

    def start(self) -> None:
        '''
        Start the worker processes.
        '''

        for worker_idx, (q, log_q) in enumerate(zip(self.queues, self.log_queues)):
            p = multiprocessing.Process(
                target=Cache.worker,
                kwargs={
                    'dataset': self.dataset,
                    'cache_dir': self.cache_dir,
                    'cache_size': self.cache_size,
                    'index_queue': q,
                    'worker_idx': worker_idx,
                    'n_workers': self.n_workers,
                    'clean_up': self.clean_up,
                    'log_queue': log_q,
                },
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def shutdown(self) -> None:
        if len(self.workers) > 0:
            for q in self.queues:
                q.put(None, block=True, timeout=None)
            for p in self.workers:
                p.join()

    @staticmethod
    def worker(
        dataset: Dataset,
        cache_dir: Path,
        cache_size: int,
        index_queue: multiprocessing.Queue,
        worker_idx: int,
        n_workers: int,
        clean_up: bool = True,
        log_queue: Optional[multiprocessing.Queue] = None,
    ) -> None:
        '''
        Cache worker. Communicates with main process via queue `queue`. Main worker process sends indices of samples to
        retrieve to queue. The worker will store these in the order in which retrieved. Duplicate indices will be dropped.
        The worker continuously loops through the local list of indices as well as cache positions, retrieves the data
        and stores it in cache.

        Args:
            dataset: Pytorch dataset to cache.
            cache_dir: Directory to use for cache.
            cache_size: Maximum number of elements stored in cache.
            index_queue: Multiprocessing queue to communicate with worker.
            worker_idx: Identifies this worker vs. others.
            n_workers: Number of workers on node.
            clean_up: Set to bool to remove all files from cache at termination.
            log_queue: Queue to return logging information to the main process.
        '''
        dataset_indices = []
        current_dataset_idx = 0
        samples_retrieved = 0
        terminate_worker = False
        # Assure worker_idx < cache_size
        if worker_idx >= cache_size:
            raise ValueError('There should not be more workers than cache slots.')
        cache_idx = worker_idx

        # Main worker loop
        while True:
            # Retrieve all elements from queue
            while True:
                try:
                    idx = index_queue.get(block=False)
                except queue.Empty:
                    break

                if idx is None:
                    terminate_worker = True
                    break
                elif idx == -2:
                    if log_queue is not None:
                        try:
                            log_queue.put(samples_retrieved, block=False)
                        except queue.Full:
                            pass
                    samples_retrieved = 0
                elif idx == -1:
                    dataset_indices = []
                    current_dataset_idx = 0
                elif idx in dataset_indices:
                    continue
                else:
                    dataset_indices.append(idx)
                    # Given that we are checking for duplicates above, we should never have too many indices.
                    # So this should never be triggered.
                    if len(dataset_indices) > len(dataset):
                        raise ValueError(
                            'Attempting to keep track of more indices than present in dataset.'
                        )

            # Check if asked to terminate
            if terminate_worker:
                logging.info('Terminating cache worker')
                break

            # Check if there is anything to query
            if len(dataset_indices) == 0:
                continue

            if current_dataset_idx >= len(dataset_indices):
                current_dataset_idx = 0
            if cache_idx >= cache_size:
                cache_idx = worker_idx

            # Retrieve and store data
            sample_idx = dataset_indices[current_dataset_idx]
            sample = dataset[sample_idx]
            file_name = cache_dir / cache_file_name(cache_idx, cache_size)

            lock_file_name = file_name.with_suffix('.lock')
            try:
                with FileLock(lock_file_name, timeout=-1):
                    torch.save(sample, file_name)
                samples_retrieved += 1
            except Timeout:
                raise RuntimeError('Blocking call leads to timeout.')
            current_dataset_idx += 1
            cache_idx += n_workers

        # Clean-up. Simply remove all files.
        if clean_up:
            for cache_idx in range(cache_size):
                file_name = cache_dir / cache_file_name(cache_idx, cache_size)
                try:
                    os.remove(file_name)
                except IOError:
                    pass
                try:
                    os.remove(file_name.with_suffix('.lock'))
                except IOError:
                    pass
