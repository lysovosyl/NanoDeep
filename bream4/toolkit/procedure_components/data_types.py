from __future__ import annotations

import threading
from typing import Generic, TypeVar

from typing_extensions import TypeAlias

K = TypeVar("K")
V = TypeVar("V")

T = TypeVar("T")


class StateMap(Generic[K, V]):
    """This is a class used for maintaining a map of up to date items
    in a thread safe way.
    Pop_all is the addition so that getting a copy of the list and emptying it is thread safe
    as it isn't guaranteed to be thread safe in traditional Python
    """

    def __init__(self):
        self._map = {}
        self._lock = threading.RLock()

    def update(self, new_states: dict[K, V]) -> None:
        """Add some channel states to the map, overwriting any channels that clash

        :param new_states: dict of channel->channel_state
        """
        with self._lock:
            self._map.update(new_states)

    def pop_all(self) -> dict[K, V]:
        """Remove everything from the map and return it

        :returns: states
        :rtype: dict

        """
        with self._lock:
            ret = {k: v for (k, v) in self._map.items()}
            self._map = {}
            return ret

    def get_all(self) -> dict[K, V]:
        """Get the current state of the state map

        :returns: states
        :rtype: dict

        """
        with self._lock:
            ret = {k: v for (k, v) in self._map.items()}
            return ret


class NotifyList(Generic[T]):
    """This is a class used for holding a list wrapped in a lock. This is
    used to wake other threads up when things are added to the list -
    Based on the condition passed to the constructor.
    :code:`len(list)` is also thread safe
    """

    def __init__(self, condition: threading.Condition):
        """Given a condition, construct a thread safe list.

        :param condition: The condition to lock/notify on when new items are available
        """
        self._list = []
        # Re-entrant lock, same thread can always 'reacquire' lock for free
        self._lock = threading.RLock()
        self._condition = condition

    def peek(self) -> T:
        """Thread safe look at the first item in the list

        :returns: First item in the list
        """
        with self._lock:
            return self._list[0]

    def pop(self) -> T:
        """Thread safe pop the first item in the list

        :returns: First item in the list
        """

        with self._lock:
            return self._list.pop(0)

    def get_copy(self) -> list[T]:
        """Thread safe copy of the list - This will not update when the list updates.
        *Only valid for shallow lists*
        :returns: Shallow copy of the list
        """

        with self._lock:
            return self._list[::]

    def append(self, item: T) -> None:
        """Add item to the list in a thread safe way. This will notify the condition
        passed in via the constructor

        :param item: Item to put to the back of the list
        """
        # When pushing, notify(wake up) listener as there is something new there
        with self._condition:
            with self._lock:
                self._list.append(item)
                self._condition.notify()

    def __len__(self) -> int:
        with self._lock:
            return len(self._list)


TimeineItem: TypeAlias = "tuple[float, V]"


class TimelineList(NotifyList[TimeineItem]):
    """An ordered thread safe notifying list"""

    def insert_ordered(self, item: TimeineItem) -> None:
        """Insert the item into the list in a sorted manner. Thread safe and notifies the condition
        If two items have the same timestamp the first one added will be earlier in the list

        :param item: Item to add in order: (timestamp, extras)
        """
        with self._condition:
            with self._lock:
                # Copied from bisect.insort_right as we don't want to descend down tuples
                # in the comparison

                low = 0
                high = len(self._list)

                while low < high:
                    mid = (low + high) // 2
                    if item[0] < self._list[mid][0]:
                        high = mid
                    else:
                        low = mid + 1

                self._list.insert(low, item)

                self._condition.notify()

    def append(self, item: TimeineItem):
        raise RuntimeError("This will break the ordered list")
