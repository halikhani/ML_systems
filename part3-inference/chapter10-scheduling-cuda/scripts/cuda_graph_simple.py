async def overlap_scheduler_loop(self):
    """SGLang's overlapped scheduling loop."""
    last_batch = None
    last_result = None
    while True:
        # Step 1: Schedule next batch (CPU)
        # This happens WHILE previous batch is computing!
        next_batch = self.get_next_batch_to_run()

        # Step 2: Launch next batch (non-blocking GPU)
        next_result = self.run_batch(next_batch)


        # Step 3: Process PREVIOUS batch results (CPU)
        # By now, previous batch is likely done
        if last_batch is not None:
            self.process_batch_result(last_batch, last_result)

        last_batch = next_batch
        last_result = next_result

"""The ultimate optimization:

CUDA Graphs for decode batches (fixed shape, repeated)
Overlap scheduling for prefill/mixed batches
FutureMap to bridge the gap"""