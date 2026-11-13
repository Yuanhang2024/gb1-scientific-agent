# 实验记录：完整保存当前状态，并追加每次操作的事件日志。
"""Single-writer JSON state and append-only operational events."""
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# 临时文件完整写入并刷新后，再替换正式状态文件，避免保存半份JSON。
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# 日志采用一行一个JSON的追加形式，按发生顺序保留调用和阶段变化。
def event(path, kind, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "kind": kind, **data},
                                ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# 同一输出目录只允许一个写入进程，避免不同进程交错覆盖实验状态。
@contextmanager
def single_writer(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".writer.lock"
    try:
        handle = lock.open("x", encoding="ascii")
    except FileExistsError as exc:
        raise RuntimeError(f"Run already locked: {lock}. After an interrupted process, verify it stopped before removing this lock.") from exc
    try:
        with handle:
            handle.write(str(os.getpid()))
            handle.flush()
            yield
    finally:
        lock.unlink(missing_ok=True)
