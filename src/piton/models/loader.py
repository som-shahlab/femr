from __future__ import annotations

import queue
import threading
import random
import piton.extension.dataloader

from typing import Optional


class Batches:
    def __init__(
        self,
        split: str,
        seed: int,
        batch_info_path: str,
        data_path: str,
        num_batch_threads: int,
        token_dropout: float = 0,
    ):
        index_queue: queue.Queue[Optional[int]] = queue.Queue(maxsize=300)

        loader = piton.extension.dataloader.BatchCreator(
            data_path, batch_info_path
        )

        def index_thread(
            index_queue, seed, total_steps, num_train_batches, num_batch_threads
        ):
            rng = random.Random(seed)
            order = None
            for step in range(total_steps):
                if step % num_train_batches == 0:
                    order = list(range(num_train_batches))
                    rng.shuffle(order)

                index_queue.put((order[step % num_train_batches], step))

            for _ in range(num_batch_threads):
                index_queue.put(None)

        batcher_thread = threading.Thread(
            target=index_thread,
            args=(
                index_queue,
                seed,
                num_train_batches,
                args.num_batch_threads,
            ),
            name="batch_thread",
            daemon=True,
        )
        batcher_thread.start()

        self.batch_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=300)

        def batch_thread(
            index_queue, batch_queue, data_path, batch_info_path, token_dropout
        ):
            thread_loader = piton.extension.dataloader.BatchCreator(
                data_path, batch_info_path, token_dropout=token_dropout
            )
            while True:
                next_item = index_queue.get()
                if next_item is None:
                    batch_queue.put(None)
                    break

                batch_index, step = next_item

                batch = thread_loader.get_batch("train", batch_index)
                batch = jax.tree_map(lambda a: jnp.array(a), batch)
                batch_queue.put((batch, step))

            batch_queue.put(None)

        batcher_threads = [
            threading.Thread(
                target=batch_thread,
                args=(
                    index_queue,
                    self.batch_queue,
                    args.data_path,
                    args.batch_info_path,
                    args.dropout,
                ),
                name="batch_thread",
                daemon=True,
            )
            for _ in range(num_batch_threads)
        ]

        for t in batcher_threads:
            t.start()

        self.remaining_threads = num_batch_threads

    def get_next(self):
        next_item = None

        while next_item is None:
            next_item = self.batch_queue.get()
            if next_item is not None:
                return next_item
            else:
                self.remaining_threads -= 1
                if self.remaining_threads == 0:
                    return None
