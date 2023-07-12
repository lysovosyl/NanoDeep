from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import os.path
import sys
import threading
import weakref
from typing import TYPE_CHECKING, Any, Optional, TextIO

import toml

if TYPE_CHECKING:
    from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

from bream4.toolkit.procedure_components.dict_utils import find_path, recursive_merge

LOG_LEVEL = os.environ.get("BREAM_LOG_LEVEL", "INFO")
BACKUPS = 30
FILE_SIZE_LIMIT = 10 * 1000 * 1000  # 10MB logs
FILENAME = "bream-0.txt"  # Will produce bream-0.txt, bream-1.txt, ...


def load_user_messages() -> dict:
    messages = {}

    if "ONT_CONFIG_DIR" in os.environ:
        config_dir = os.environ["ONT_CONFIG_DIR"]
        filenames = [
            os.path.join(config_dir, "shared", "user_messages.toml"),
            os.path.join(config_dir, "production", "user_messages.toml"),
        ]

        for fn in filenames:
            try:
                with open(fn, "r", encoding="utf-8") as toml_file:
                    result = toml.load(toml_file)
                    recursive_merge(messages, result)
            except Exception:  # nosec
                pass

    return messages


class RotatingFileHandlerNoRoll(logging.handlers.RotatingFileHandler):
    def rotation_filename(self, filename: str) -> str:
        """Return the new log name from the filename

        This is never called for the initial log (bream-0.txt)

        :param filename: str of the form /path/to/bream-0.txt.1 bream-0.txt.2...
        :returns: new str filename /path/to/bream-1.txt, path/to/bream-2.txt,
        """

        filename, rotations = filename.rsplit(".", 1)

        new_filename = filename[: -len(FILENAME)] + FILENAME.replace("0", rotations)
        return new_filename

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        """
        Either:
        * The file doesn't yet exist, so rollover
        * the file is too big so we should rollover
        * we haven't yet rotated so we should
        """

        # If we don't have a file yet, then make sure we rotate to get a stream
        if not os.path.isfile(self.baseFilename):
            return True

        # Else if the file is too big, also rollover
        # Note that shouldRollover actually create the file if it isn't there
        if super().shouldRollover(record):
            return True

        # Otherwise return if we haven't rotated yet. If we haven't rotated yet we want to
        # as this means we are on a new protocol run
        return not SharedLoggerInfo().rotated

    def doRollover(self) -> None:
        """Override the rollover method for windows. If a handle is already on then
        just don't rotate. Windows log files could get big but it's that or crash the run
        """

        try:
            if self.stream:
                faulthandler.disable()
                self.stream.close()
                self.stream = None  # type: ignore Base class evaluates as not optional even though it is
            os.rename(self.baseFilename, self.baseFilename)
            super().doRollover()
        except (NameError, OSError):
            pass

        SharedLoggerInfo().set_rotated(True)

    def _open(self) -> TextIO:
        """Override the open method to make sure that the faulthandler is looking at the correct stream.
        Can't do during rotation as we have delay=True meaning file handle won't open until at least one log written"""

        handle = super()._open()
        faulthandler.enable(file=handle)
        return handle


# Used to share values between logger instances
class SharedLoggerInfo:
    log_dir = os.path.expanduser("~")
    log_path = os.path.join(log_dir, FILENAME)
    handler = RotatingFileHandlerNoRoll(
        log_path,
        backupCount=BACKUPS,
        delay=True,
        maxBytes=FILE_SIZE_LIMIT,
        encoding="utf-8",
    )

    formatter = logging.Formatter("[%(name)s:%(lineno)4s] - %(asctime)15s - %(levelname)8s - %(message)s")

    device_ref: Optional[weakref.ReferenceType] = None
    rotated = False
    user_messages: Optional[dict] = None
    lock = threading.RLock()

    def __enter__(self):
        self.lock.acquire()
        if not self.user_messages:
            self.set_user_messages(load_user_messages())
        return self

    def __exit__(self, *args):
        self.lock.release()

    @classmethod
    def generate_handler(cls) -> None:
        cls.handler = RotatingFileHandlerNoRoll(cls.log_path, backupCount=BACKUPS, delay=True, maxBytes=FILE_SIZE_LIMIT)
        cls.handler.setLevel(LOG_LEVEL)
        cls.handler.setFormatter(cls.formatter)

    @classmethod
    def set_user_messages(cls, messages: dict) -> None:
        cls.user_messages = messages

    @classmethod
    def set_device(cls, device: "BaseDeviceInterface") -> None:
        if cls.device_ref:
            # If already got a ref, things are set up.
            # If we have an out of date ref, update it
            if not cls.device_ref():
                cls.device_ref = weakref.ref(device)
            return

        cls.device_ref = weakref.ref(device)
        cls.log_dir = device.get_logs_directory()
        cls.log_path = os.path.join(cls.log_dir, FILENAME)
        cls.generate_handler()

    @classmethod
    def set_rotated(cls, rotated: bool) -> None:
        cls.rotated = rotated

    @classmethod
    def reset(cls) -> None:
        cls.log_dir = os.path.expanduser("~")
        cls.log_path = os.path.join(cls.log_dir, FILENAME)
        cls.generate_handler()

        cls.device_ref = None
        cls.rotated = False


class BreamLogger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.name = name

        self.setLevel(LOG_LEVEL)

        sys.excepthook = self.log_uncaught_exception

    def log_uncaught_exception(self, exc_type, exc_value, exc_traceback) -> None:
        if not issubclass(exc_type, KeyboardInterrupt):
            super().error("", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def handle(self, record: logging.LogRecord) -> None:
        """Overriding Logger's handle method to make sure we always use the
        most up to date logger handler so any loggers obtained before a device
        handler can still output to the correct location

        :param record: record to emit

        """

        with SharedLoggerInfo() as info:
            info.handler.handle(record)

    def _resolve_message(self, msg: str, params: Optional[dict[str, Any]]) -> str:
        with SharedLoggerInfo() as info:
            if not info.user_messages:
                return msg

            # Expand message
            conf_message = find_path(msg, info.user_messages)
            if conf_message and params:
                resolved_msg = conf_message["msg"].format(**params)
            elif conf_message:
                resolved_msg = conf_message["msg"]
            else:
                resolved_msg = msg

        return resolved_msg

    def log_to_gui(self, msg: str, params: Optional[dict[str, Any]] = None) -> None:
        """
        New logging handler that logs to both the GUI and the log file using
        loggers' internal mechanisms

        :param self: required for logging.Logger
        :param msg:  The log message that will be written to the log and
            displayed in the GUI
        :param params: parameters to resolve the string
        """

        resolved_msg = self._resolve_message(msg, params)

        with SharedLoggerInfo() as info:
            super().log(self.level, f"{resolved_msg} (Params: {params})")

            # If not instantiated (None) or instantiated but deleted (lambda: None)
            if info.device_ref and info.device_ref():
                info.device_ref().print_log_to_gui(resolved_msg, identifier=msg, extra_data=params)  # type: ignore
            else:
                msg = (
                    "Need to call attach_device_to_logger(device) with a device to log to gui."
                    "Only writing non-gui logs"
                )
                super().warning(msg)

    def error(self, msg: str, *args, params: Optional[dict[str, Any]] = None, **kwargs) -> None:
        """
        New logging handler that logs to both the GUI and the log file using
        loggers' internal mechanisms

        :param self: required for logging.Logger
        :param msg:  The error message that will be written to the log and
            displayed in the GUI
        :param params: parameters to resolve the string
        """

        resolved_msg = self._resolve_message(msg, params)

        with SharedLoggerInfo() as info:
            super().error(f"{resolved_msg} (Params: {params})", *args, **kwargs)

            # If not instantiated (None) or instantiated but deleted (lambda: None)
            if info.device_ref and info.device_ref():
                info.device_ref().print_error_to_gui(resolved_msg, identifier=msg, extra_data=params)  # type: ignore
            else:
                msg = (
                    "Need to call attach_device_to_logger(device) with a device to log to gui."
                    "Only writing non-gui logs"
                )
                super().warning(msg)


def attach_device_to_logger(device: "BaseDeviceInterface") -> None:
    """
    Attaches the device to the loggers so log_to_gui and log file paths
    are sensible
    """
    with SharedLoggerInfo() as info:
        info.set_device(device)


logging.Logger.manager.setLoggerClass(BreamLogger)  # type: ignore
