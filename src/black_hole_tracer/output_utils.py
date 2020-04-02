import curses
import multiprocessing as multi
import time


# command line output
class Outputter:
    def __init__(self, num_threads):
        self.message = {}
        self._num_threads = num_threads
        self.queue = multi.Queue()
        self.stdscr = curses.initscr()
        curses.noecho()

        for i in range(self._num_threads):
            self.message[i] = "..."
        self.message[-1] = "..."

    def name(self, num):
        if num == -1:
            return "M"
        else:
            return str(num)

    def doprint(self):
        for i in range(self._num_threads + 1):
            self.stdscr.addstr(
                i,
                0,
                self.name(i - 1) + "] " + self.message[i - 1],
            )
        self.stdscr.refresh()

    def parsemessages(self):
        doref = False

        while not self.queue.empty():
            i, m = self.queue.get()
            self.setmessage(m, i)
            doref = True

        if doref:
            self.doprint()

    def setmessage(self, mess, i):
        self.message[i] = mess.ljust(60)

    def __del__(self):
        try:
            curses.echo()
            curses.endwin()
            print("\n" * (self._num_threads + 1))
        except:
            pass


def format_time(secs):
    if secs < 60:
        return "%d s" % secs
    if secs < 60 * 3:
        return "%d m %d s" % divmod(secs, 60)
    return "%d min" % (secs / 60)
