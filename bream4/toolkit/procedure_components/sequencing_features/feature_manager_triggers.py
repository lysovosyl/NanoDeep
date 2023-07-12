import logging
from typing import Callable, Optional


class GlobalTrigger(object):
    """
    Global trigger class, designed to trigger the exit of a feature manager loop when triggered
    The class has a simple boolean attribute that allows you to test the defined triggers to determine which one has
    triggered the exit of the manager loop. allowing to specific sections of code to subsequently execute.

    """

    def __init__(self, enable: bool = False) -> None:
        """

        :param enable: used to identify the the trigger that has caused the feature manager to exit and control the
        execution of new code
        """
        self.logger = logging.getLogger(__name__)
        self.enable = enable

    def execute(self, exit_manager_sleep: Optional[Callable[[], None]] = None) -> None:
        """
        executed on triggering by the feature manager, will throw an exception if an exit feature is not passed.
        :param exit_manager_sleep:
        :return:
        """
        if exit_manager_sleep is not None:
            self.logger.info("Triggering exit of feature manager")
            exit_manager_sleep()
            self.enable = True
        else:
            raise RuntimeError("No exit handler provided to global trigger")
