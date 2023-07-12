__version__ = "7.1.3"
__DESCRIPTION__ = "Tools to run ONT hardware"

import sys
import threading

# Ensure logging handler attached
import bream4.utility.logging_utils  # noqa: F401


def install_thread_excepthook():
    """
    Fix in Python 3.8

    Workaround for sys.excepthook thread issue (https://bugs.python.org/issue1230540)
    From:
    http://spyced.blogspot.com/2007/06/workaround-for-sysexcepthook-bug.html
    """

    init_old = threading.Thread.__init__

    def init(self, *args, **kwargs):
        init_old(self, *args, **kwargs)
        run_old = self.run

        def run_with_except_hook(*args, **kw):
            try:
                run_old(*args, **kw)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:  # noqa: E722
                sys.excepthook(*sys.exc_info())  # type: ignore ex_info will give values as in an exception

        self.run = run_with_except_hook

    threading.Thread.__init__ = init


install_thread_excepthook()
