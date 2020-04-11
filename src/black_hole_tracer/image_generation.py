""" Use multiple process to generate an image. """
# Standard Imports
import ctypes
import multiprocessing as multi
import random
import sys
import time

# 3rd Party Imports
import numpy as np

# Local Imports
from .utils.output import Outputter


class ScheduledImageGenerator:
    """ Generate an image using multiple processes """

    def __init__(self, resolution, num_processes, chunk_size, shuffle=True):
        """ Initialize a scheduled ray tracer.

        Args:
            resolution (tuple(int, int)):
                The x, and y dimensions of the image to generate.

            num_processes (int): The number of processes to run.

            chunk_size (int): The number of pixels to trace for each chunk.

        Keyword Args:
            shuffle (bool): Shuffle the pixels that go into chunks if True.
        """
        self._resolution = resolution
        self._num_processes = num_processes
        self._shuffle = shuffle
        self._start_time = 0

        self._num_pixels = self._resolution[0] * self._resolution[1]
        pixel_indices = np.arange(0, self._num_pixels)

        if self._shuffle:
            np.random.shuffle(pixel_indices)

        chunks = np.array_split(pixel_indices, self._num_pixels / chunk_size + 1)

        self._colour_buffer_preproc_shared = multi.Array(ctypes.c_float, self._num_pixels * 3)
        self._colour_buffer_preproc = self._to_numpy_array(
            self._colour_buffer_preproc_shared,
            self._num_pixels,
        )

        # Shuffle chunks to equalize load
        random.shuffle(chunks)

        # partition chunk list in schedules for single threads
        self._schedules = []

        # from http://stackoverflow.com/questions/2659900/python-slicing-a-list-into-n-nearly-equal-length-partitions
        ########## Change this after all else tested
        q, r = divmod(len(chunks), self._num_processes)
        indices = [q * i + min(i, r) for i in range(self._num_processes + 1)]

        for i in range(self._num_processes):
            self._schedules.append(chunks[indices[i]:indices[i + 1]])
        ##########

        self._iter_counters = np.zeros(self._num_processes).astype(int) # Add to rk4 or delete
        self._chunk_counters = np.zeros(self._num_processes).astype(int)
        self._killers = np.zeros(self._num_processes).astype(bool)

        self._output = None


    @property
    def colour_buffer_preproc(self):
        """"""
        return self._colour_buffer_preproc

    @property
    def start_time(self):
        """"""
        return self._start_time

    @staticmethod # Try without static
    def _to_numpy_array(mp_arr, num_pixels):
        """
        """
        array = np.frombuffer(mp_arr.get_obj(), dtype=np.float32)
        array.shape = (num_pixels, 3)
        return array



    def _show_progress(self, message_string, index):
        """
        """
        return self._output.queue.put(
            (
                index,
                "Chunk {}/{}, {}".format(
                    self._chunk_counters[index],
                    len(self._schedules[index]),
                    message_string.ljust(30),
                )
            )
        )

    def generate_image(self):
        """
        """
        self._output = Outputter(self._num_processes)

        # Start clock to time raytracing
        self._start_time = time.time()

        # Multiprocessing
        process_list = []
        for process_num in range(self._num_processes):
            process = multi.Process(
                target=self._scheduled_generation,
                args=(
                    process_num,
                    self._schedules[process_num],
                )
            )
            process_list.append(process)

        ##### Try moving this ^^
        for process in process_list:
            process.start()

        try:
            refresh_counter = 0
            while True:
                refresh_counter += 1
                time.sleep(0.1)

                self._output.parsemessages()

                self._output.setmessage("Idle.", -1)

                all_done = True
                for process in process_list:
                    if process.is_alive():
                        all_done = False
                        break

                if all_done:
                    break

        except KeyboardInterrupt:
            for process_num in range(self._num_processes):
                self._killers[process_num] = True
            sys.exit()

        del self._output

    def _scheduled_generation(self, process_num, schedule):
        """
        """
