from __future__ import annotations

import multiprocessing
import pickle
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar

import meds_reader

A = TypeVar("A")

WorkEntry = Tuple[Callable[[Iterable[meds_reader.Patient]], Any], List[int]]


def _runner(
    database: meds_reader.PatientDatabase,
    input_queue: multiprocessing.Queue[Optional[WorkEntry]],
    result_queue: multiprocessing.Queue[Any],
) -> None:
    while True:
        next_work = input_queue.get()
        if next_work is None:
            break

        map_func, patient_ids = next_work

        map_func = pickle.loads(map_func)

        result = map_func(database[patient_id] for patient_id in patient_ids)
        result_queue.put(result)


class Pool:
    def __init__(self, database: meds_reader.PatientDatabase, num_threads: int) -> None:
        self._all_patient_ids = list(database)
        self._num_threads = num_threads

        if num_threads != 1:
            self._processes = []
            mp = multiprocessing.get_context("spawn")

            self._input_queue: multiprocessing.Queue[Optional[WorkEntry]] = mp.Queue()
            self._result_queue: multiprocessing.Queue[Any] = mp.Queue()

            for _ in range(num_threads):
                process = mp.Process(
                    target=_runner,
                    kwargs={"database": database, "input_queue": self._input_queue, "result_queue": self._result_queue},
                )
                process.start()
                self._processes.append(process)
        else:
            self._database = database

    def map(
        self, map_func: Callable[[Iterable[meds_reader.Patient]], A], patient_ids: Optional[List[int]] = None
    ) -> Iterator[A]:
        """Apply the provided map function to the database"""

        if patient_ids is None:
            patient_ids = self._all_patient_ids

        if self._num_threads != 1:
            patients_per_part = (len(patient_ids) + len(self._processes) - 1) // len(self._processes)

            num_work_entries = 0

            map_func_p = pickle.dumps(map_func)

            for i in range(len(self._processes)):
                patient_ids_for_thread = patient_ids[i * patients_per_part : (i + 1) * patients_per_part]

                if len(patient_ids_for_thread) == 0:
                    continue

                num_work_entries += 1
                self._input_queue.put((map_func_p, patient_ids_for_thread))

            return (self._result_queue.get() for _ in range(num_work_entries))
        else:
            return (map_func(self._database[patient_id] for patient_id in patient_ids),)

    def terminate(self) -> None:
        """Close the pool"""
        if self._num_threads != 1:
            for _ in self._processes:
                self._input_queue.put(None)
            for process in self._processes:
                process.join()

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()


__all__ = ["Pool"]
