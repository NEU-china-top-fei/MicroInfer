"""Synchronous text/token generation over the MicroInfer engine."""

import json
from dataclasses import replace
from pathlib import Path
from threading import Lock

from microinfer.config import QwenConfig, SampleParameter
from microinfer.config.engine import EngineConfig
from microinfer.engine.contracts import RequestOutput
from microinfer.engine.processor import InputProcessor, OutputProcessor


class LLM:
    def __init__(self, engine, input_processor, output_processor):
        self.engine = engine
        self.input_processor = input_processor
        self.output_processor = output_processor
        self._generation_lock = Lock()

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer,
        *,
        block_size=16,
        num_blocks=64,
        engine_config=None,
        debug_mode=None,
        use_compile=None,
        fuse_weights=True,
        use_cuda_graph=None,
        compile_ops=None,
        use_speculative=None,
        draft_model=None,
        speculator=None,
    ):
        from microinfer.engine.blockmanager import BlockManager
        from microinfer.engine.llm_engine import LLMEngine
        from microinfer.engine.scheduler import Scheduler
        from microinfer.layer.sampler import Sampler
        from microinfer.runner import ModelRunner

        config = engine_config or EngineConfig(
            max_model_len=min(model.config.max_pos_embed, block_size * num_blocks),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        overrides = {}
        if debug_mode is not None:
            overrides["debug_mode"] = debug_mode
        if use_compile is not None:
            overrides["use_compile"] = use_compile
        if use_cuda_graph is not None:
            overrides["use_cuda_graph"] = use_cuda_graph
        if compile_ops is not None:
            overrides["compile_ops"] = compile_ops
        if use_speculative is not None:
            overrides["use_speculative"] = use_speculative
        elif draft_model is not None:
            overrides["use_speculative"] = True
        if overrides:
            config = replace(config, **overrides)
        if config.max_model_len > model.config.max_pos_embed:
            raise ValueError("engine context exceeds model position limit")
        if fuse_weights:
            fuse = getattr(model, "fuse_weights", None)
            if fuse is not None:
                fuse()
        if config.compile_ops:
            model.enable_compile_ops()
        runner = ModelRunner(
            model.eval(),
            None,
            Sampler(validate=config.debug_mode),
            block_size=block_size,
            device=model.embedding.weight.device,
            debug_mode=config.debug_mode,
            use_compile=config.use_compile,
            use_cuda_graph=config.use_cuda_graph,
            max_num_seqs=config.max_num_seqs,
        )
        runner.allocate_kv_cache(num_blocks)
        scheduler = Scheduler(config, BlockManager(num_blocks, block_size))
        if speculator is None and draft_model is not None and config.use_speculative:
            from microinfer.engine.speculative import DraftModelSpeculator

            speculator = DraftModelSpeculator(
                draft_model,
                max_drafts=config.speculative_max_drafts,
                vocab_size=model.config.vocab_size,
            )
        return cls(
            LLMEngine(scheduler, runner, speculator=speculator),
            InputProcessor(
                tokenizer,
                max_model_len=scheduler.max_length,
                vocab_size=model.config.vocab_size,
            ),
            OutputProcessor(tokenizer),
        )

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        device=None,
        dtype=None,
        block_size=16,
        num_blocks=64,
        engine_config=None,
        debug_mode=None,
        use_compile=None,
        fuse_weights=True,
        use_cuda_graph=None,
        compile_ops=None,
        use_speculative=None,
        draft_model_path=None,
        draft_layers=None,
        draft_num_blocks=None,
    ):
        """Load a local Qwen3 directory. Never downloads or executes remote code."""
        import torch
        from transformers import AutoTokenizer

        from microinfer.model.loader import load_model
        from microinfer.model.qwen3 import QwenModel

        path = Path(path)
        settings = json.loads((path / "config.json").read_text())
        config = QwenConfig.from_hf(settings)
        device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        if isinstance(dtype, str):
            dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype]
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unsupported model dtype")
        # Construct directly in the target dtype/device. Building in FP32 and
        # converting doubles peak memory and OOMs larger checkpoints.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device(device):
                model = QwenModel(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        load_model(model, path)
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        draft_model = None
        native_speculator = None
        if draft_model_path is not None:
            if draft_layers is not None:
                from microinfer.engine.native_draft import NativeDraftSpeculator
                from microinfer.model.draft import load_pruned_draft

                draft_model = load_pruned_draft(
                    draft_model_path,
                    num_layers=draft_layers,
                    dtype=dtype,
                    device=device,
                )
                draft_compile_ops = (
                    compile_ops
                    if compile_ops is not None
                    else bool(engine_config and engine_config.compile_ops)
                )
                if draft_compile_ops:
                    draft_model.enable_compile_ops()
                native_speculator = NativeDraftSpeculator(
                    draft_model,
                    block_size=block_size,
                    num_blocks=draft_num_blocks or num_blocks,
                    max_drafts=(
                        engine_config.speculative_max_drafts
                        if engine_config is not None
                        else 8
                    ),
                    max_num_seqs=(
                        engine_config.max_num_seqs if engine_config is not None else 8
                    ),
                    use_cuda_graph=bool(use_cuda_graph),
                )
            else:
                from transformers import AutoModelForCausalLM

                draft_model = AutoModelForCausalLM.from_pretrained(
                    draft_model_path,
                    local_files_only=True,
                    trust_remote_code=False,
                    dtype=dtype,
                )
                draft_model.to(device)
        return cls.from_model(
            model,
            tokenizer,
            block_size=block_size,
            num_blocks=num_blocks,
            engine_config=engine_config,
            debug_mode=debug_mode,
            use_compile=use_compile,
            fuse_weights=fuse_weights,
            use_cuda_graph=use_cuda_graph,
            compile_ops=compile_ops,
            use_speculative=use_speculative,
            draft_model=draft_model,
            speculator=native_speculator,
        )

    def generate(self, prompts, sampling_params=None, use_tqdm=True):
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("this LLM already has an active generate call")
        seqs, admitted = [], []
        progress = None
        try:
            if not self.engine.is_finished():
                raise RuntimeError("engine has pending requests")
            if isinstance(prompts, str):
                prompts = [prompts]
            prompts = list(prompts)
            if sampling_params is None:
                sampling_params = SampleParameter()
            params = (
                [sampling_params] * len(prompts)
                if isinstance(sampling_params, SampleParameter)
                else list(sampling_params)
            )
            if len(params) != len(prompts):
                raise ValueError("one sampling parameter set is required per prompt")
            seqs = [
                self.input_processor.process(prompt, param)
                for prompt, param in zip(prompts, params)
            ]
            for seq in seqs:
                self.engine.add_request(seq)
                admitted.append(seq)
            if use_tqdm and seqs:
                from tqdm.auto import tqdm

                progress = tqdm(total=len(seqs), desc="Generating")
            results = {}
            while not self.engine.is_finished():
                outputs, _ = self.engine.step()
                for seq_id, tokens in outputs:
                    decoded = self.output_processor.process(
                        RequestOutput(seq_id, tuple(tokens), True)
                    )
                    results[seq_id] = {"text": decoded.text, "token_ids": list(tokens)}
                    if progress is not None:
                        progress.update(1)
            return [results[s.seq_id] for s in seqs]
        except Exception:
            for seq in admitted:
                self.engine.abort(seq.seq_id)
            raise
        finally:
            if progress is not None:
                progress.close()
            self._generation_lock.release()
