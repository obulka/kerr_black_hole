import curses
import multiprocessing as multi
import sys
import time


class Outputter:
    """ Class for multiprocessed command line output. """

    def __init__(self, num_threads):
        """
        """
        self._message = {}
        self._num_threads = num_threads
        self._queue = multi.Queue()
        self.stdscr = curses.initscr()
        curses.noecho()

        for i in range(self._num_threads):
            self._message[i] = "..."
        self._message[-1] = "..."

    @property
    def queue(self):
        """"""
        return self._queue

    def name(self, num):
        """
        """
        if num == -1:
            return "M"
        else:
            return str(num)

    def doprint(self):
        """
        """
        for i in range(self._num_threads + 1):
            self.stdscr.addstr(
                i,
                0,
                self.name(i - 1) + "] " + self._message[i - 1],
            )
        self.stdscr.refresh()

    def parsemessages(self):
        """
        """
        doref = False

        while not self._queue.empty():
            i, m = self._queue.get()
            self.setmessage(m, i)
            doref = True

        if doref:
            self.doprint()

    def setmessage(self, mess, i):
        """
        """
        self._message[i] = mess.ljust(60)

    def __del__(self):
        try:
            curses.echo()
            curses.endwin()
            print("\n" * (self._num_threads + 1))
        except:
            pass


def print_help_and_exit():
    print("Use the -h or --help switch for more help.")
    print("Exiting...")
    sys.exit()
