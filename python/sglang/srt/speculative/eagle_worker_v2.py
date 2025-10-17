import logging
from typing import List, Optional

import torch
from torch.cuda import Stream as CudaStream

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.build_eagle_tree import TreeMaskMode
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    build_tree_kernel_efficient_tmp,
    fill_accepted_out_cache_loc,
    fill_new_verified_id,
    select_top_k_tokens_tmp,
)
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.utils.common import empty_context, fast_topk, next_power_of_2

logger = logging.getLogger(__name__)


class EAGLEWorkerV2(EAGLEWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            nccl_port,
            target_worker,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        self.tree_mask_mode = TreeMaskMode.FULL_MASK
        self.plan_stream: CudaStream = torch.get_device_module(self.device).Stream()
        # TODO(lsyin): potential bugs with a separate plan stream
        self.plan_stream_ctx = torch.cuda.stream(self.plan_stream)
        self.plan_stream_ctx = empty_context()

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if model_worker_batch.forward_mode.is_decode():
            # FIXME(lsyin): why shall we use spec_info for both draft and verify?
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            assert draft_input.is_draft_input()
            verify_input: EagleVerifyInput = self.draft(model_worker_batch)
            assert verify_input.is_verify_input()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch, draft_input.allocate_lens)
            return batch_output
        else:
            # Target prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            batch_output.next_draft_input = self.forward_draft_extend(
                model_worker_batch,
                batch_output.logits_output.hidden_states,
                batch_output.next_token_ids,
            )
            return batch_output

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
            self.req_to_token_pool,
            model_worker_batch,
            self.cuda_graph_runner,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Run draft
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch,
            )
        else:
            self.draft_attn_backend.init_forward_metadata(forward_batch)
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient_tmp(
            draft_input.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            model_worker_batch.seq_lens,
            model_worker_batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens_tmp(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_model_runner.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
            self._detect_nan_if_needed(logits_output)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def verify(
        self,
        batch: ModelWorkerBatch,
        pre_draft_allocate_lens: torch.Tensor,
    ):
        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        seq_lens_backup = batch.seq_lens
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = (
                verify_input.prepare_for_v2_verify(
                    self.req_to_token_pool,
                    batch,
                    self.target_worker,
                )
            )

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Run target verify batch in the main compute stream
        forward_batch_output = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Sample
        self._detect_nan_if_needed(logits_output)
        (
            predict,
            accept_length,
            accept_index,
        ) = verify_input.sample(batch, logits_output)
        new_seq_lens = seq_lens_backup + accept_length
        verify_done = torch.cuda.Event()

        # Move the accepted tokens to the target KV cache locations
        batch.seq_lens = seq_lens_backup
        self.move_accepted_tokens_to_target_kvcache(
            batch,
            accept_index,
            accept_length,
        )

        verify_done.record()

        all_verified_id = predict[accept_index]
        verified_id = torch.empty_like(accept_length, dtype=torch.int32)
        fill_new_verified_id[(bs,)](
            all_verified_id,
            accept_length,
            verified_id,
            self.speculative_num_draft_tokens,
        )

        # Batch 2: Draft extend
        draft_input = EagleDraftInput(
            hidden_states=logits_output.hidden_states,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + accept_length
            - 1
        )

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                predict,
                self.speculative_num_draft_tokens,
                self.draft_model_runner,
            )

        if self.plan_stream:
            torch.cuda.current_stream().wait_stream(self.plan_stream)

        # Run draft extend batch in the main compute stream
        draft_logits_output = self.draft_model_runner.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        probs = torch.softmax(draft_logits_output.next_token_logits, dim=-1)
        ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
        ret_hidden_states = draft_logits_output.hidden_states

        # Since seq_lens_backup's tensor is allocated in another stream, we
        # need record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        seq_lens_backup.record_stream(torch.cuda.current_stream())

        # Construct the return values
        next_draft_input = EagleDraftInput(
            topk_p=ret_topk_p,
            topk_index=ret_topk_index,
            hidden_states=ret_hidden_states,
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=pre_draft_allocate_lens,
            verify_done=verify_done,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            last_batch_allocate_lens=pre_draft_allocate_lens,
        )

    def forward_draft_extend(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        pt = 0
        for i, extend_len in enumerate(batch.extend_seq_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], next_token_ids[i].reshape(1))
            )
            pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            verified_id=next_token_ids,
            new_seq_lens=batch.seq_lens,
            allocate_lens=batch.seq_lens,
        )
        batch.spec_info = next_draft_input

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_model_runner)
        logits_output, _ = self.draft_model_runner.forward(forward_batch)

        # Update spec_info for the next draft step
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        next_draft_input.topk_p, next_draft_input.topk_index = fast_topk(
            probs, self.topk, dim=-1
        )
        next_draft_input.hidden_states = logits_output.hidden_states
        return next_draft_input

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            accept_length: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + accept_length,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def _detect_nan_if_needed(self, logits_output: LogitsProcessorOutput):
        if self.enable_nan_detection:
            logits = logits_output.next_token_logits
            if torch.any(torch.isnan(logits)):
                logger.error("Detected errors during sampling! NaN in the logits.")
                raise ValueError("Detected errors during sampling! NaN in the logits.")


def free_spec_dec_tokens_page_size_1(
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: TokenToKVPoolAllocator,
    req: Req,
    allocate_len: int,
    new_seq_len: int,
    page_size: int,
):
    # FIXME(lsyin): move this function elsewhere

    # free extra allocated tokens
    if new_seq_len is None:
        # True only for overlap eagle and the current batch is decode. This seq will be part of the decode, so the final iteration's allocation is not used (i.e. this case).
        start_len = allocate_len - EagleDraftInput.ALLOC_LEN_PER_DECODE
    else:
        # True for 1) non-overlap; 2) overlap eagle and the current batch is prefill. This seq will not run extra iteration, so start_lens is passed in.
        start_len = new_seq_len

    # Debug logging to understand which case we're in
    logger.info(
        f"free_spec_dec_tokens_page_size_1: case analysis - "
        f"new_seq_len={new_seq_len}, start_len={start_len}, allocate_len={allocate_len}, page_size={page_size}"
    )

    if page_size == 1:
        # For page_size = 1, free tokens directly
        indices_to_free = req_to_token_pool.req_to_token[req.req_pool_idx][
            start_len:allocate_len
        ]
        token_to_kv_pool_allocator.free(indices_to_free)
    else:
        # For page_size > 1, we need to handle page alignment carefully
        # to avoid conflicts with radix cache truncation

        req_token_indices = req_to_token_pool.req_to_token[req.req_pool_idx]
        tokens_to_free = req_token_indices[start_len:allocate_len]

        if len(tokens_to_free) == 0:
            return

        if new_seq_len is None:
            # Case 1: Radix cache truncation happens BEFORE this function
            # We need to be careful not to double-free tokens that radix cache already handled

            # Calculate the actual sequence length that radix cache sees
            actual_seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1

            # Radix cache truncates to page-aligned length
            page_aligned_seq_len = actual_seq_len // page_size * page_size

            # Only free tokens that are beyond what radix cache can see
            # and align to page boundaries to avoid partial page issues
            if start_len < page_aligned_seq_len:
                # Some tokens might be handled by radix cache, be conservative
                # Only free tokens from the next page boundary after what radix cache sees
                next_page_boundary = (
                    (page_aligned_seq_len + page_size - 1) // page_size
                ) * page_size
                if next_page_boundary < allocate_len:
                    tokens_to_free = req_token_indices[next_page_boundary:allocate_len]
                    if len(tokens_to_free) > 0:
                        token_to_kv_pool_allocator.free(tokens_to_free)
            else:
                # All tokens are beyond radix cache visibility, safe to free
                # But still align to page boundaries
                page_aligned_start = (
                    (start_len + page_size - 1) // page_size
                ) * page_size
                if page_aligned_start < allocate_len:
                    tokens_to_free = req_token_indices[page_aligned_start:allocate_len]
                    token_to_kv_pool_allocator.free(tokens_to_free)
        else:
            # Case 2: Radix cache truncation happens AFTER this function
            # We need to ensure we don't free tokens that radix cache will handle later

            # Calculate what radix cache will see and free
            all_token_len = (
                len(req.origin_input_ids) + len(req.output_ids) - 1
            )  # Exclude last token
            actual_kv_len = all_token_len - 1  # For EAGLE, bigram keys
            page_aligned_kv_len = actual_kv_len // page_size * page_size

            # Debug logging to understand the calculation
            logger.info(
                f"free_spec_dec_tokens_page_size_1: calculation details - "
                f"origin_input_ids_len={len(req.origin_input_ids)}, "
                f"output_ids_len={len(req.output_ids)}, "
                f"all_token_len={all_token_len}, "
                f"actual_kv_len={actual_kv_len}, "
                f"page_aligned_kv_len={page_aligned_kv_len}, "
                f"allocate_len={allocate_len}"
            )

            # Radix cache will free kv_indices[page_aligned_kv_len:]
            # This means radix cache will free tokens from page_aligned_kv_len to all_token_len

            # We should only free tokens that are beyond what radix cache will handle
            # Since radix cache handles up to all_token_len, we should only free beyond that
            if all_token_len < allocate_len:
                tokens_to_free = req_token_indices[all_token_len:allocate_len]
                if len(tokens_to_free) > 0:
                    # For page_size > 1, we need to align to page boundaries based on actual token indices
                    # We need to ensure we only free complete pages, not partial pages

                    # Get the actual token indices we want to free
                    actual_indices_to_free = tokens_to_free

                    # Calculate page boundaries based on actual indices
                    # We need to be very careful about page alignment to avoid conflicts with radix cache
                    if len(actual_indices_to_free) > 0:
                        first_token_idx = actual_indices_to_free[0].item()
                        last_token_idx = actual_indices_to_free[-1].item()

                        # Calculate page-aligned boundaries
                        first_page = first_token_idx // page_size
                        last_page = last_token_idx // page_size

                        # Check if radix cache will free any tokens in the same pages
                        # Radix cache frees kv_indices[page_aligned_kv_len:] where page_aligned_kv_len=1544
                        # We need to get the actual token indices that radix cache will free
                        radix_tokens_to_free = req_token_indices[
                            page_aligned_kv_len:all_token_len
                        ]

                        logger.info(
                            f"free_spec_dec_tokens_page_size_1: radix analysis - "
                            f"page_aligned_kv_len={page_aligned_kv_len}, all_token_len={all_token_len}, "
                            f"radix_tokens_to_free={radix_tokens_to_free.tolist() if len(radix_tokens_to_free) > 0 else []}"
                        )

                        if len(radix_tokens_to_free) > 0:
                            radix_first_token = radix_tokens_to_free[0].item()
                            radix_last_token = radix_tokens_to_free[-1].item()
                            radix_start_page = radix_first_token // page_size
                            radix_end_page = radix_last_token // page_size
                        else:
                            radix_start_page = radix_end_page = -1

                        logger.info(
                            f"free_spec_dec_tokens_page_size_1: page analysis - "
                            f"our_tokens={actual_indices_to_free.tolist()}, "
                            f"our_pages={first_page} to {last_page}, "
                            f"radix_pages={radix_start_page} to {radix_end_page}"
                        )

                        # Check for overlap with radix cache pages and only free non-overlapping pages
                        # Handle non-contiguous tokens that may span multiple pages
                        tokens_to_free_final = []

                        for token_idx in actual_indices_to_free:
                            token_page = token_idx.item() // page_size

                            # Check if this specific token's page overlaps with radix cache pages
                            if radix_start_page <= token_page <= radix_end_page:
                                # This token's page overlaps with radix cache, skip it
                                logger.info(
                                    f"free_spec_dec_tokens_page_size_1: skipping token {token_idx.item()} "
                                    f"in page {token_page} - overlaps with radix cache pages {radix_start_page}-{radix_end_page}"
                                )
                            else:
                                # This token's page doesn't overlap, safe to free
                                logger.info(
                                    f"free_spec_dec_tokens_page_size_1: keeping token {token_idx.item()} "
                                    f"in page {token_page} - no overlap with radix cache pages {radix_start_page}-{radix_end_page}"
                                )
                                tokens_to_free_final.append(token_idx)

                        if len(tokens_to_free_final) > 0:
                            tokens_to_free_tensor = torch.stack(tokens_to_free_final)
                            logger.info(
                                f"free_spec_dec_tokens_page_size_1: freeing non-overlapping tokens - "
                                f"tokens={tokens_to_free_tensor.tolist()}"
                            )
                            token_to_kv_pool_allocator.free(tokens_to_free_tensor)
                        else:
                            logger.info(
                                f"free_spec_dec_tokens_page_size_1: no non-overlapping tokens to free"
                            )
                    else:
                        logger.info(
                            f"free_spec_dec_tokens_page_size_1: no tokens to free"
                        )
