import atexit
import dataclasses
import re
import shutil
import tempfile
from typing import Dict, Tuple

from torch.utils.benchmark import Language

from tasks.definitions import FLAT_TASKS
import tasks.jit
from tasks.setup import SetupMap
from tasks.spec import Mode, TaskSpec, TimerSpec


@dataclasses.dataclass()
class Task:
    name: str
    sub_tasks: Dict[Mode, TimerSpec]

    def __init__(self, key: Tuple[str, ...], task_spec: TaskSpec, working_dir: str):
        self.name = re.sub(r'[^a-z_]', '_', '_'.join(key).lower())
        self.sub_tasks = {
            Mode.PY: TimerSpec(
                stmt=task_spec.py_stmt,
                setup=SetupMap[task_spec.setup][Language.PYTHON],
                language=Language.PYTHON,
            ),

            Mode.CPP: TimerSpec(
                stmt=task_spec.cpp_stmt,
                setup=SetupMap[task_spec.setup][Language.CPP],
                language=Language.CPP,
            ),
        }

        model_path = tasks.jit.generate_torchscript_file(
            task_spec,
            self.name,
            working_dir=working_dir,
        )

        if model_path is not None:
            for mode, language in ((Mode.PY_TS, Language.PYTHON), (Mode.CPP_TS, Language.CPP)):
                ts_stmt, ts_setup = tasks.jit.make_stmt_and_setup(
                    language=language,
                    setup=SetupMap[task_spec.setup][language],
                    model_path=model_path,
                    torchscript_args=task_spec.torchscript_args,
                )
                self.sub_tasks[mode] = TimerSpec(
                    stmt=ts_stmt,
                    setup=ts_setup,
                    language=language,
                )

    @staticmethod
    def key_to_name(key: Tuple[str, ...]):
        return re.sub(r'[^a-z_]', '_', '_'.join(key).lower())


@dataclasses.dataclass()
class Tasks:
    entries: Dict[Tuple[str, ...], Task]

    def __init__(self):
        self._working_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, self._working_dir)

        self.entries = {
            k: Task(k, v, self._working_dir)
            for k, v in FLAT_TASKS.items()
        }
