import argparse
import io
import pickle
import sys
import traceback
from typing import List, Optional, Union

from torch.utils.benchmark import CallgrindStats, Measurement

from runner.core import (
    CALLGRIND_NUMBER, MeasurementType, MIN_RUN_TIME,
    WorkerInput, WorkerOutput)
from tasks.spec import make_timer


def run(worker_input: WorkerInput) -> WorkerOutput:
    results: Union[List[Measurement], List[CallgrindStats]] = []
    try:
        for timer_spec in worker_input.tasks:
            timer = make_timer(timer_spec)
            if worker_input.measurement_type == MeasurementType.CALLGRIND:
                results.append(timer.collect_callgrind(
                    number=CALLGRIND_NUMBER, collect_baseline=False))
            else:
                assert worker_input.measurement_type == MeasurementType.WALL_TIME
                results.append(timer.blocked_autorange(min_run_time=MIN_RUN_TIME))

    except KeyboardInterrupt:
        # Runner process sent SIGINT.
        sys.exit()

    except:
        # If a worker fails, we want to ship the Exception back to the caller
        # rather than raising in the worker.
        f = io.StringIO()
        traceback.print_exc(file=f)
        return WorkerOutput(results=(), e_str=f.getvalue())

    return WorkerOutput(tuple(results))


def main(communication_file: str) -> None:
    with open(communication_file, 'rb') as f:
        worker_input = pickle.load(f)
        assert isinstance(worker_input, WorkerInput)

    worker_output: WorkerOutput = run(worker_input)
    with open(communication_file, 'wb') as f:
        pickle.dump(worker_output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--communication_file', type=str)
    communication_file = parser.parse_args().communication_file
    main(communication_file)
