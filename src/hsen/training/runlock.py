"""One trainer per run directory.

The feature cache has had an extraction lock since before this phase, for a
reason the project already learned the hard way: two writers on one directory
strand each other's index entries.  Training output had no such guard, and it
has now cost a run.

What happened.  A long ablation job was reported killed; its child process was
not, and a resumed job ran alongside it.  Both wrote the same
``results/hsen/<experiment>/<profile>/`` tree for nearly an hour.  The artefacts
that came out are individually well-formed and jointly incoherent -- ``best.pt``
records epoch 10 with a validation score that appears nowhere in its own history
-- which is the worst shape a corruption can take, because nothing errors and
the numbers look ordinary.

So: a run directory is claimed for the life of a training run, and a second
trainer is refused rather than allowed to interleave.  The lock records the pid,
the host and the start time, so a stale lock can be judged rather than guessed
at, and ``--force-lock`` exists for when the holder is confirmed gone.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path


class RunDirectoryBusy(RuntimeError):
    """Raised when another trainer already holds this run directory."""


class RunLock:
    """An advisory lock over one run directory."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "training.lock"
        self._held = False

    def owner(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"unreadable": True}

    @staticmethod
    def _process_alive(pid: int) -> bool | None:
        """Whether a pid is running, or ``None`` when that cannot be determined.

        ``None`` rather than a guess: reporting a dead holder as alive strands
        the directory, and reporting a live one as dead invites exactly the
        concurrent write this lock exists to prevent.

        Windows needs its own path.  ``os.kill(pid, 0)`` is a POSIX idiom that
        CPython does not emulate there -- against a live process it raises
        ``SystemError``, which is not in the exception set anyone writes for
        this check.  The result is that reporting the lock's *owner* crashes,
        turning a clear refusal into a traceback.  ``OpenProcess`` is the
        supported question to ask.
        """
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return None

        if platform.system() == "Windows":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True          # exists, owned by someone else
        except Exception:
            return None
        return True

    def acquire(self, force: bool = False) -> "RunLock":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            owner = self.owner()
            alive = self._process_alive(owner.get("pid", -1))
            status = {True: "still running", False: "no longer running",
                      None: "status unknown"}[alive]
            raise RunDirectoryBusy(
                f"{self.run_dir} is already claimed by another training run.\n"
                f"  pid     {owner.get('pid')} ({status})\n"
                f"  host    {owner.get('host')}\n"
                f"  started {owner.get('started_at')}\n"
                f"  command {owner.get('command')}\n\n"
                f"Two trainers writing one run directory produce artefacts that "
                f"are individually valid and jointly wrong -- a best.pt whose "
                f"recorded score is absent from its own history. This project has "
                f"already had that happen once.\n\n"
                f"If the holder is confirmed gone, pass --force-lock. Otherwise "
                f"use a different --run_name."
            )
        payload = {
            "pid": os.getpid(),
            "host": platform.node(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": " ".join(os.sys.argv[:4]),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._held = True
        return self

    def release(self) -> None:
        if self._held and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        self._held = False

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
