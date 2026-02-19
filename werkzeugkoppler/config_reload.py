"""Watchdog-based config reload loop."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Awaitable, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


def watchdog_path_matches_config(path: str | Path | None, watch_name: str) -> bool:
    """Return true when filesystem event path points to the watched config file name."""
    if not path:
        return False
    return Path(path).name == watch_name


class ConfigReloadWatcher:
    """Watch one config file and invoke async reload callback on changes."""

    def __init__(
        self,
        *,
        config_file: Path,
        on_reload: Callable[[Path], Awaitable[None]],
        logger: logging.Logger,
    ) -> None:
        self._config_file = config_file
        self._on_reload = on_reload
        self._log = logger
        self._mtime: float | None = self._current_mtime()

    def _current_mtime(self) -> float | None:
        return self._config_file.stat().st_mtime if self._config_file.exists() else None

    async def reload_if_changed(self, *, force: bool = False) -> bool:
        """Reload config when mtime changed (or force=True)."""
        mtime = self._current_mtime()
        if not force and (mtime is None or mtime == self._mtime):
            return False

        self._log.info("Configuration change detected at %s, reloading...", self._config_file)
        await self._on_reload(self._config_file)
        self._mtime = mtime
        self._log.info("Configuration reloaded successfully")
        return True

    async def _watchdog_reload_loop(self) -> None:
        """Watch config file changes via watchdog and reload immediately."""
        loop = asyncio.get_running_loop()
        changed = asyncio.Event()
        watch_dir = self._config_file.parent.resolve()
        watch_name = self._config_file.name

        class _ConfigEventHandler(FileSystemEventHandler):
            """Signal async reload loop when config file changes."""

            def _touch(self, path: str | Path | None) -> None:
                if not watchdog_path_matches_config(path, watch_name):
                    return
                loop.call_soon_threadsafe(changed.set)

            def on_modified(self, event: FileSystemEvent) -> None:
                if getattr(event, "is_directory", False):
                    return
                self._touch(getattr(event, "src_path", None))

            def on_created(self, event: FileSystemEvent) -> None:
                if getattr(event, "is_directory", False):
                    return
                self._touch(getattr(event, "src_path", None))

            def on_moved(self, event: FileSystemEvent) -> None:
                if getattr(event, "is_directory", False):
                    return
                self._touch(getattr(event, "src_path", None))
                self._touch(getattr(event, "dest_path", None))

            def on_deleted(self, event: FileSystemEvent) -> None:
                if getattr(event, "is_directory", False):
                    return
                self._touch(getattr(event, "src_path", None))

        observer = Observer()
        handler = _ConfigEventHandler()
        observer.schedule(handler, str(watch_dir), recursive=False)
        observer.start()
        try:
            while True:
                await changed.wait()
                changed.clear()
                try:
                    await self.reload_if_changed(force=True)
                except Exception as exc:
                    self._log.warning("Configuration reload failed, keeping current config: %s", exc)
        finally:
            observer.stop()
            # join() is blocking; call in thread to keep event loop responsive.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(observer.join, 2.0)

    async def run_forever(self) -> None:
        """Run watchdog loop continuously and auto-restart on watcher failure."""
        while True:
            try:
                await self._watchdog_reload_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.warning("watchdog config watcher failed (%s), retrying in 1s", exc)
                await asyncio.sleep(1.0)
