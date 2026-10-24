from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

type Process_fn[In, Out] = Callable[[list[In]], Awaitable[list[Out]]]

logger = logging.getLogger(__name__)


class BatcherError(Exception):
    """Basic class of batcher errors."""


class QueueFullError(BatcherError):
    """Batcher error for a full queue. Will become a 503."""


class BatcherClosedError(BatcherError):
    """Batcher error for a closed or closing batcher."""


class BatchSizeMismatchError(BatcherError):
    """Batcher error for if the process function returned a number of outputs
    diffrent than the number of inputs."""


class Batcher[In, Out]:
    """A batch goes to processing when the batch is full or when
    it waited for `max_wait_ms` since the arrival of the first element."""

    def __init__(
        self,
        process_fn: Process_fn[In, Out],
        *,
        max_batch_size: int,
        max_wait_ms: float,
        max_queue_size: int,
    ) -> None:

        self.process_fn = process_fn
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        self._max_batch_size = max_batch_size
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        self._max_wait_ms = max_wait_ms
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0")
        self._max_queue_size = max_queue_size
        self.q = asyncio.Queue(maxsize=max_queue_size)
        self._closed = True
        self._worker_task = None

    async def start(self) -> None:
        """Starts the the worker as a background task"""

        if self._worker_task is not None:
            raise asyncio.InvalidStateError("The worker is already started")
        self._worker_task = asyncio.create_task(self._run())
        self._closed = False

    async def stop(self) -> None:
        """Clean stop of the worker.
        If the worker is stopped while there remain futures, there are
        ended with an exception."""

        if self._closed:
            return
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._fail_remaining(BatcherClosedError("Batcher closed"))

    async def __aenter__(self) -> Batcher[In, Out]:
        """Starts the batcher."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Stops the batcher."""
        await self.stop()



    async def submit(self, item: In) -> Out:
        """Submits an item and waits for the result of the future."""
        
        if self._closed:
            raise BatcherClosedError("Batcher closed")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self.q.put_nowait((item, future))
        except asyncio.QueueFull as e:
            raise QueueFullError() from e
        return await future

    def qsize(self) -> int:
        """Returns the number of elements waiting in the queue."""
        return self.q.qsize()

    def _end_futures(self, batch, e) -> None:
        """Ends waiting futures with a chosen exception"""
        for _, fut in batch:
            if not fut.done():
                fut.set_exception(e)


    async def _run(self) -> None:
        """Worker loop: repeats "collect a batch, process it" until it stops.        """
        while True:
            batch = None
            try:
                batch = await self._collect_batch()
                await self._process_batch(batch)
            except asyncio.CancelledError:
                if batch is not None:
                    self._end_futures(batch, BatcherClosedError("Batcher closed"))
                raise
            except Exception as e:
                if batch is not None:
                    self._end_futures(batch, e)
                logger.exception("DynamicBatcher._run") 


    async def _collect_batch(self):
        """Collects the batch: picks elements in the queue until it has reached 
        `max_batch_size` or waited `max_wait_ms` since the arrival of the first element.
        """

        loop = asyncio.get_running_loop()
        batch = []
        item, fut = await self.q.get()
        batch.append((item, fut))
        deadline = loop.time() + self._max_wait_ms/1000
        restant = deadline - loop.time()
        try: 
            while len(batch) < self._max_batch_size and restant > 0:
                if self.q.qsize() > 0:
                    item, fut = self.q.get_nowait()
                else:
                    item, fut = await asyncio.wait_for(self.q.get(), timeout=restant)
                batch.append((item, fut))
                restant = deadline - loop.time()
        except TimeoutError:
            pass
        except asyncio.CancelledError:
            self._end_futures(batch, BatcherClosedError("Batcher closed"))
            raise
        return batch


    async def _process_batch(self, batch) -> None:
        """Calls the processing function then ends futures with the correct output
        or with an exception."""

        items, futures = [], []
        for item, fut in batch:
            if not fut.done():
                items.append(item)
                futures.append(fut)
        if len(items) == 0:
            return  
        try:      
            res = await self.process_fn(items)
        except Exception as e:
            self._end_futures(batch, e)
            logger.exception("DynamicBatcher._process_batch")
            return
        if len(res) != len(items):
            raise BatchSizeMismatchError(f"process results size is {len(res)} != number of items {len(items)}")
        l = zip(res, futures)
        for out, fut in l:
            if not fut.done():
                fut.set_result(out)


    def _fail_remaining(self, exc: BaseException) -> None:
        """Empties the queue and ends every remaining future with the exception."""
        
        batch = []
        while self.q.qsize() > 0:
            item, fut = self.q.get_nowait()
            if not fut.done():
                batch.append((item, fut))
        self._end_futures(batch, exc)
