#!/usr/bin/env python
# -*- coding:utf-8 -*-

""" # Updated in 21/03/10 """
import os, signal, logging, sys
from time import sleep, time
from multiprocessing import JoinableQueue, Event, Process
from queue import Empty
from typing import Optional

def do_task(data1: int, data2: int, data3: int, logger=None) -> int:
    data = data1 + data2 + data3
    logger.info("Task data : [%d]", data)
    return data


def worker(func, q: JoinableQueue, stop_event: Event, logger=None):
    logger.info("Starting worker...")
    while 1:
        if stop_event.is_set():
            logger.info("Worker exiting because of stop_event")
            break

        try:
            args = q.get(timeout=.01)
        except Empty:
            continue

        if args is None:
            logger.info("Worker exiting because of None on queue")
            q.task_done()
            break

        # Do the task
        try:
            func(*args[0])
        except:
            logger.exception("Failed to process args %s", args)
        finally:
            q.task_done()

def run(
        func,
        data,
        n_workers: int = 2,
        n_tasks: int = 10,
        max_queue_size: int = 1,
        grace_period: int = 2,
        kill_period: int = 30,
        interrupt: Optional[int] = None,
        logger=None
) -> None:
    """
    Run a process pool of workers.

    Args:
        n_workers: Start this many processes
        n_tasks: Launch this many tasks
        max_queue_size: If queue exceeds this size, block when putting items on the queue
        grace_period: Send SIGINT to processes if they don't exit within this time after SIGINT/SIGTERM
        kill_period: Send SIGKILL to processes if they don't exit after this many seconds
        interrupt: If given, send signal SIGTERM to itself after queueing this many tasks
        :param func:
    """

    q = JoinableQueue(maxsize=max_queue_size)
    stop_event = Event()

    def handler(signalname):
        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    procs = []
    for i in range(n_workers):
        p = Process(name=f"Worker-{i:02d}", daemon=True, target=worker, args=(func, q, stop_event, logger))
        procs.append(p)
        p.start()
    try:
        # Put tasks on queue
        for i_task in range(n_tasks):
            if interrupt and i_task == interrupt:
                os.kill(os.getpid(), signal.SIGTERM)

            logger.info("Put [{}] on queue".format(data[i_task]))
            q.put([data[i_task]])

        # Wait until all tasks are processed
        q.join()
    except KeyboardInterrupt:
        logger.warning("Caught KeyboardInterrupt! Setting stop event...")
    finally:
        stop_event.set()
        t = time()
        # Send SIGINT if process doesn't exit quickly enough, and kill it as last resort
        # .is_alive() also implicitly joins the process (good practice in linux)
        alive_procs = [p for p in procs if p.is_alive()]
        while alive_procs:
            alive_procs = [p for p in procs if p.is_alive()]
            if time() > t + grace_period:
                for p in alive_procs:
                    os.kill(p.pid, signal.SIGINT)
                    logger.warning("Sending SIGINT to %s", p)
            elif time() > t + kill_period:
                for p in alive_procs:
                    logger.warning("Sending SIGKILL to %s", p)
                    # Queues and other inter-process communication primitives can break when
                    # process is killed, but we don't care here
                    p.kill()
            sleep(.01)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    # Make sure to log process/thread name
    logging.basicConfig(level=logging.INFO, format="%(processName)s %(relativeCreated)d %(message)s")

    data = []
    for i in range(10):
        data.append([i, 0, 0])
    run(func=do_task, data=data,
        n_workers=2, n_tasks=len(data), max_queue_size=len(data))