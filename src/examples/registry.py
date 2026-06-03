from __future__ import annotations

from typing import Dict

from examples.add2x2 import Add2x2Task
from examples.apply2x2 import Apply2x2Task
from examples.addition import AdditionTask
from examples.addition2 import Addition2Task
from examples.mnist_sudoku import MNISTSudokuTask
from examples.member3 import Member3Task
from examples.member5 import Member5Task

from examples.common import NeSyTask

TASKS: Dict[str, NeSyTask] = {
    AdditionTask.name: AdditionTask(),
    Addition2Task.name: Addition2Task(),
    Add2x2Task.name: Add2x2Task(),
    Apply2x2Task.name: Apply2x2Task(),
    Member3Task.name: Member3Task(),
    Member5Task.name: Member5Task(),
    MNISTSudokuTask.name: MNISTSudokuTask(),
}
