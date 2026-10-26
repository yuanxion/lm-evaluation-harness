import logging
import json
from pathlib import Path
from importlib.util import find_spec
from math import ceil

import numpy as np
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger(__name__)


@register_model("openvino-mx")
class OpenVINOMX(LM):
    """OpenVINO backend using VLMPipeline with optional LLMPipeline fallback.

    For multimodal generation tasks, `Instance.args` is
    `(context, gen_kwargs, aux_arguments)` and `aux_arguments["visual"]`
    contains image objects.
    For text-only generation tasks, this backend can fall back to LLMPipeline
    when no visual input is provided.
    """

    MULTIMODAL = True

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    @staticmethod
    def _parse_llamacpp_device(device: str | None) -> dict:
        if device is None:
            return {}

        device_normalized = str(device).strip()
        device_upper = device_normalized.upper()

        if device_upper == "CPU":
            return {}

        if device_upper.startswith("VULKAN"):
            suffix = device_normalized[len("Vulkan") :]
            gpu_index = 0 if suffix == "" else int(suffix)
            return {"main_gpu": gpu_index}

        if device_upper.startswith("GPU"):
            suffix = device_normalized[len("GPU") :].lstrip(".")
            gpu_index = 0 if suffix == "" else int(suffix)
            return {"main_gpu": gpu_index}

        return {}

    @staticmethod
    def _parse_ov_config(raw_cfg) -> dict:
        if raw_cfg is None:
            return {}
        if isinstance(raw_cfg, dict):
            return raw_cfg
        if isinstance(raw_cfg, str):
            raw_cfg = raw_cfg.strip()
            if not raw_cfg:
                return {}
            if raw_cfg.startswith("{"):
                try:
                    parsed = json.loads(raw_cfg)
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    eval_logger.warning("Failed to parse inline ov_config JSON: %s", raw_cfg)
                    return {}
            path = Path(raw_cfg)
            if path.is_file():
                try:
                    parsed = json.loads(path.read_text())
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    eval_logger.warning("Failed to parse ov_config file as JSON: %s", path)
                    return {}
        return {}

    def __init__(
        self,
        pretrained: str,
        device: str = "CPU",
        max_new_tokens: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self._device = device
        self.default_max_new_tokens = int(max_new_tokens)
        self.use_llamacpp = self._as_bool(kwargs.get("llamacpp", False))
        self.llamacpp_chat = self._as_bool(kwargs.get("llamacpp_chat", False))
        self.llamacpp_n_ctx = kwargs.get("llamacpp_n_ctx")
        self.llamacpp_n_gpu_layers = kwargs.get("llamacpp_n_gpu_layers")
        self.rng_seed = int(kwargs["seed"]) if kwargs.get("seed") is not None else None
        self.llama = None
        self.pipe = None
        self.text_pipe = None
        self.ov = None
        self.GenerationConfig = None

        if not self.use_llamacpp:
            if not find_spec("openvino"):
                raise ModuleNotFoundError(
                    "package `openvino` is not installed. Please install it first."
                )

            if not find_spec("pipeline"):
                raise ModuleNotFoundError(
                    "Could not import `pipeline`. Ensure your VLMPipeline module is importable."
                )

            import openvino as ov
            try:
                from openvino.genai import GenerationConfig, LLMPipeline, VLMPipeline
            except ModuleNotFoundError as err:
                if err.name != "openvino.genai":
                    raise
                from openvino_genai import GenerationConfig, LLMPipeline, VLMPipeline

            self.ov = ov
            self.GenerationConfig = GenerationConfig
            pretrained_path = Path(self.pretrained)
            if not pretrained_path.is_file():
                self.pipe = VLMPipeline(self.pretrained, self._device)
        self.allow_text_only = self._as_bool(kwargs.get("allow_text_only", True))
        self.prompt_suffix = str(kwargs.get("prompt_suffix", ""))
        self.chat_template_path = kwargs.get("chat_template_path")
        self.chat_template = None
        self._configured_chat_template_pipes = set()
        if self.chat_template_path:
            template_path = Path(self.chat_template_path)
            if not template_path.is_file():
                raise FileNotFoundError(
                    f"chat_template_path does not exist: {template_path}"
                )
            self.chat_template = template_path.read_text(encoding="utf-8")
            if not self.chat_template.strip():
                raise ValueError(f"chat template is empty: {template_path}")

        self.ov_config = self._parse_ov_config(kwargs.get("ov_config"))

        # Parse apply_chat_template: default True unless specified False in kwargs or ov_config
        self.apply_chat_template = kwargs.get("apply_chat_template")
        if self.apply_chat_template is None and "apply_chat_template" in self.ov_config:
            self.apply_chat_template = self._as_bool(self.ov_config["apply_chat_template"])
        if self.apply_chat_template is None:
            self.apply_chat_template = True
        else:
            self.apply_chat_template = self._as_bool(self.apply_chat_template)

        # Parse enable_thinking: default False unless specified True in kwargs or ov_config
        self.enable_thinking = kwargs.get("enable_thinking")
        if self.enable_thinking is None and "enable_thinking" in self.ov_config:
            self.enable_thinking = self._as_bool(self.ov_config["enable_thinking"])
        if self.enable_thinking is None:
            self.enable_thinking = False
        else:
            self.enable_thinking = self._as_bool(self.enable_thinking)

        if self.use_llamacpp:
            if not find_spec("llama_cpp"):
                raise ModuleNotFoundError(
                    "package `llama_cpp` is not installed. Please install llama-cpp-python first."
                )
            from llama_cpp import Llama

            llama_kwargs = self._parse_llamacpp_device(self._device)
            if self.llamacpp_n_ctx is not None:
                llama_kwargs["n_ctx"] = int(self.llamacpp_n_ctx)
            if self.llamacpp_n_gpu_layers is not None:
                llama_kwargs["n_gpu_layers"] = int(self.llamacpp_n_gpu_layers)
            if self.rng_seed is not None:
                llama_kwargs["seed"] = self.rng_seed
            self.llama = Llama(self.pretrained, **llama_kwargs)
        else:
            self.text_pipe = LLMPipeline(self.pretrained, self._device)
        self.min_image_side = int(
            kwargs.get("min_image_side", self._infer_min_image_side(self.pretrained))
        )

        self.debug_image_dir = Path(kwargs.get("debug_image_dir", "openvino_mx_debug_images"))
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        self._image_dump_counter = 0

        self.debug_prompt_log_path = kwargs.get("debug_prompt_log_path")
        self.debug_failure_log_path = kwargs.get("debug_failure_log_path")
        self.debug_prompt_log_max = int(kwargs.get("debug_prompt_log_max", 8))
        self._prompt_log_count = 0
        self.debug_token_log_path = kwargs.get("debug_token_log_path")
        self.debug_token_log_max = int(kwargs.get("debug_token_log_max", 8))
        self._token_log_count = 0

    @staticmethod
    def _infer_min_image_side(pretrained: str) -> int:
        config_path = Path(pretrained) / "preprocessor_config.json"
        default_side = 32
        if not config_path.is_file():
            return default_side

        try:
            cfg = json.loads(config_path.read_text())
            patch_size = int(cfg.get("patch_size", 0))
            merge_size = int(cfg.get("merge_size", 0))
            if patch_size > 0 and merge_size > 0:
                return patch_size * merge_size
        except Exception as err:  # noqa: BLE001
            eval_logger.warning(
                "Failed to parse %s for min image side: %s. Falling back to %d.",
                config_path,
                err,
                default_side,
            )

        return default_side

    def _ensure_min_image_side(self, image_np: np.ndarray, image_idx: int) -> np.ndarray:
        if image_np.ndim < 2:
            return image_np

        height, width = image_np.shape[:2]
        if height >= self.min_image_side and width >= self.min_image_side:
            return image_np

        scale = max(self.min_image_side / max(height, 1), self.min_image_side / max(width, 1))
        new_height = max(self.min_image_side, int(ceil(height * scale)))
        new_width = max(self.min_image_side, int(ceil(width * scale)))

        from PIL import Image

        resized = Image.fromarray(image_np.astype(np.uint8)).resize(
            (new_width, new_height), Image.BILINEAR
        )
        eval_logger.warning(
            "Upscaled tiny image %d from %dx%d to %dx%d to satisfy OpenVINO preprocessor minimum side.",
            image_idx,
            width,
            height,
            new_width,
            new_height,
        )
        return np.array(resized)

    def _save_debug_jpg(self, image) -> None:
        save_path = self.debug_image_dir / f"image_{self._image_dump_counter:06d}.jpg"
        self._image_dump_counter += 1

        if hasattr(image, "convert") and hasattr(image, "save"):
            image.convert("RGB").save(save_path, format="JPEG", quality=95)
            return

        image_np = np.array(image)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        elif image_np.ndim == 3 and image_np.shape[-1] == 4:
            image_np = image_np[..., :3]

        from PIL import Image

        Image.fromarray(image_np.astype(np.uint8)).save(
            save_path, format="JPEG", quality=95
        )

    def _extract_prompt(self, context) -> str:
        if isinstance(context, str):
            return context

        if isinstance(context, dict):
            for key in ("prompt", "text"):
                value = context.get(key)
                if isinstance(value, str):
                    return value

            content = context.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        if item_type in {"text", "input_text"} and isinstance(
                            item.get("text"), str
                        ):
                            parts.append(item["text"])
                        elif isinstance(item.get("content"), str):
                            parts.append(item["content"])
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join([p for p in parts if p]).strip()

        if isinstance(context, (list, tuple)):
            parts = []
            for item in context:
                extracted = self._extract_prompt(item)
                if extracted:
                    parts.append(extracted)
            return "\n".join(parts).strip()

        return str(context)
    def _images_to_tensors(self, aux_arguments: dict | None):
        if not aux_arguments:
            return None

        visuals = aux_arguments.get("visual")
        if not visuals:
            return None

        tensors = []
        for image_idx, image in enumerate(visuals):
            # self._save_debug_jpg(image)

            if isinstance(image, self.ov.Tensor):
                image_np = np.array(image.data)
            else:
                # Normalize to 3-channel RGB (some datasets provide RGBA or grayscale).
                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                image_np = np.array(image)

            if image_np.ndim == 4 and image_np.shape[0] == 1:
                image_np = image_np[0]

            # Convert CHW to HWC when needed.
            if image_np.ndim == 3 and image_np.shape[0] in (1, 3, 4) and image_np.shape[-1] not in (1, 3, 4):
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.ndim == 3 and image_np.shape[-1] == 1:
                image_np = np.repeat(image_np, 3, axis=-1)
            elif image_np.ndim == 3 and image_np.shape[-1] == 4:
                image_np = image_np[..., :3]

            image_np = self._ensure_min_image_side(image_np, image_idx)

            tensors.append(self.ov.Tensor(image_np))

        return tensors

    @staticmethod
    def _apply_stop(text: str, until: list[str] | None) -> str:
        if not until:
            return text
        out = text
        for term in until:
            if term:
                out = out.split(term)[0]
        return out

    @staticmethod
    def _extract_generated_text(output) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        texts = getattr(output, "texts", None)
        if texts:
            return texts[0]
        if isinstance(output, (list, tuple)) and output:
            return OpenVINOMX._extract_generated_text(output[0])
        return str(output)

    def _write_token_log(self, payload: dict) -> None:
        if not self.debug_token_log_path or self._token_log_count >= self.debug_token_log_max:
            return
        try:
            token_log_path = Path(self.debug_token_log_path)
            token_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(token_log_path, "a", encoding="utf-8") as token_file:
                token_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._token_log_count += 1
        except Exception as err:
            eval_logger.warning("Failed to write debug_token_log_path: %s", err)

    def _write_llamacpp_token_log(
        self, raw_prompt: str, rendered_prompt: str, token_ids: list[int], text: str,
        generation_kwargs: dict, generation_contract: dict, enable_thinking: bool | None,
    ) -> None:
        self._write_token_log(
            {
                "raw_prompt": raw_prompt,
                "rendered_prompt": rendered_prompt,
                "input_token_ids": token_ids,
                "generated_token_ids": list(
                    self.llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)
                ),
                "generated_token_ids_source": "re-encoded llama.cpp completion text; create_completion does not expose sampled token IDs",
                "generated_text": text,
                "generation_kwargs": generation_kwargs,
                "generation_contract": generation_contract,
                "enable_thinking": enable_thinking,
                "apply_chat_template": self.llamacpp_chat,
                "chat_template_path": self.chat_template_path,
                "tokenizer_source": "llama.cpp GGUF tokenizer",
                "tokenizer_details": {
                    "implementation": f"{type(self.llama).__module__}.{type(self.llama).__qualname__}.tokenize",
                    "special_token_parsing": True,
                    "add_bos": False,
                },
            }
        )

    def _generate_token_trace(self, prompt: str, generation_config):
        tokenizer = self.text_pipe.get_tokenizer()
        encoded = tokenizer.encode(prompt, add_special_tokens=True)
        result = self.text_pipe.generate(encoded, generation_config)
        generated_ids = [int(token_id) for token_id in result.tokens[0]]
        return encoded, generated_ids, tokenizer.decode(result.tokens[0]), tokenizer

    def _generation_contract(self, gen_kwargs: dict, default_max_new_tokens: int, until) -> dict:
        stop_strings = sorted(str(item) for item in (until or []) if item)
        return {
            "max_new_tokens": int(gen_kwargs.get("max_gen_toks", default_max_new_tokens)),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
            "top_p": float(gen_kwargs.get("top_p", 1.0)),
            "top_k": int(gen_kwargs.get("top_k", 0)),
            "min_p": float(gen_kwargs.get("min_p", 0.0)),
            "presence_penalty": float(gen_kwargs.get("presence_penalty", 0.0)),
            "repetition_penalty": float(gen_kwargs.get("repetition_penalty", 1.0)),
            "do_sample": self._as_bool(gen_kwargs.get("do_sample", gen_kwargs.get("temperature", 0) > 0)),
            "stop_strings": stop_strings,
            "seed": self.rng_seed,
        }

    @staticmethod
    def _llm_generate_kwargs(gen_kwargs: dict, default_max_new_tokens: int) -> dict:
        kwargs = {
            "max_new_tokens": int(gen_kwargs.get("max_gen_toks", default_max_new_tokens))
        }
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if key in gen_kwargs:
                kwargs[key] = gen_kwargs[key]
        if "do_sample" in gen_kwargs:
            kwargs["do_sample"] = gen_kwargs["do_sample"]
        elif gen_kwargs.get("temperature", 0) > 0:
            kwargs["do_sample"] = True
        until = gen_kwargs.get("until")
        if until:
            if isinstance(until, str):
                until = [until]
            kwargs["stop_strings"] = set(until)
        return kwargs

    def _format_chat_prompt(
        self, active_pipe, raw_prompt: str, enable_thinking: bool | None = None
    ) -> tuple[str, bool]:
        if not self.apply_chat_template:
            return raw_prompt, False

        if self.chat_template is not None:
            pipe_id = id(active_pipe)
            if pipe_id not in self._configured_chat_template_pipes:
                tokenizer = getattr(active_pipe, "get_tokenizer", lambda: None)()
                set_chat_template = getattr(tokenizer, "set_chat_template", None)
                if not callable(set_chat_template):
                    raise RuntimeError(
                        "chat_template_path was provided, but the active OpenVINO "
                        "pipeline tokenizer cannot set a chat template."
                    )
                set_chat_template(self.chat_template)
                self._configured_chat_template_pipes.add(pipe_id)

        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if thinking is not None:
            tokenizer = getattr(active_pipe, "get_tokenizer", lambda: None)()
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_template):
                raise RuntimeError(
                    "enable_thinking was configured, but the OpenVINO tokenizer "
                    "cannot apply a chat template with extra context."
                )
            try:
                formatted = apply_template(
                    [{"role": "user", "content": raw_prompt}],
                    add_generation_prompt=True,
                    extra_context={"enable_thinking": self._as_bool(thinking)},
                )
            except Exception as err:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to apply chat template with enable_thinking="
                    f"{self._as_bool(thinking)}: {err}"
                ) from err
            return formatted, False

        return raw_prompt, True

    @staticmethod
    def _llamacpp_generate_kwargs(gen_kwargs: dict, default_max_new_tokens: int) -> dict:
        kwargs = {
            "max_tokens": int(gen_kwargs.get("max_gen_toks", default_max_new_tokens)),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
        }
        if "top_p" in gen_kwargs:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if "top_k" in gen_kwargs:
            kwargs["top_k"] = int(gen_kwargs["top_k"])
        if "min_p" in gen_kwargs:
            kwargs["min_p"] = float(gen_kwargs["min_p"])
        if "presence_penalty" in gen_kwargs:
            kwargs["presence_penalty"] = float(gen_kwargs["presence_penalty"])
        if "repetition_penalty" in gen_kwargs:
            kwargs["repeat_penalty"] = float(gen_kwargs["repetition_penalty"])
        return kwargs

    def _llamacpp_generate(
        self,
        prompt: str,
        until: list[str] | None,
        gen_kwargs: dict,
        enable_thinking: bool | None = None,
    ) -> str:
        if self.llama is None:
            raise RuntimeError("openvino-mx llama-cpp mode is enabled but model is not initialized")

        llama_kwargs = self._llamacpp_generate_kwargs(gen_kwargs, self.default_max_new_tokens)
        if until:
            llama_kwargs["stop"] = until

        if self.llamacpp_chat and self.chat_template is not None:
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter

            bos_token = self.llama.detokenize([self.llama.token_bos()]).decode("utf-8", errors="ignore")
            eos_token = self.llama.detokenize([self.llama.token_eos()]).decode("utf-8", errors="ignore")
            if not eos_token:
                eos_token = "<|im_end|>"
            formatter = Jinja2ChatFormatter(
                self.chat_template, eos_token=eos_token, bos_token=bos_token
            )
            formatted = formatter(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=enable_thinking,
            )
            token_ids = self.llama.tokenize(
                formatted.prompt.encode("utf-8"), add_bos=False, special=True
            )
            self._last_llamacpp_trace = {
                "rendered_prompt": formatted.prompt,
                "input_token_ids": [int(token_id) for token_id in token_ids],
            }
            stops = [s for s in (until or []) if s]
            stops.extend([s for s in (formatted.stop or []) if s])
            if stops:
                llama_kwargs["stop"] = list(dict.fromkeys(stops))
            response = self.llama.create_completion(
                prompt=token_ids,
                **llama_kwargs,
            )
            return str(response.get("choices", [{}])[0].get("text", ""))

        if self.llamacpp_chat:
            if enable_thinking is not None:
                raise ValueError(
                    "This llama-cpp-python version cannot pass enable_thinking "
                    "to the embedded GGUF chat template. Configure chat_template_path."
                )
            response = self.llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **llama_kwargs,
            )
            self._last_llamacpp_trace = {
                "rendered_prompt": prompt,
                "input_token_ids": [
                    int(token_id)
                    for token_id in self.llama.tokenize(
                        prompt.encode("utf-8"), add_bos=False, special=True
                    )
                ],
            }
            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return str(content)

        response = self.llama.create_completion(prompt=prompt, **llama_kwargs)
        self._last_llamacpp_trace = {
            "rendered_prompt": prompt,
            "input_token_ids": [
                int(token_id)
                for token_id in self.llama.tokenize(
                    prompt.encode("utf-8"), add_bos=False, special=True
                )
            ],
        }
        return str(response.get("choices", [{}])[0].get("text", ""))

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        if not requests:
            return []

        results = []
        iterator = tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Running OpenVINO VLM generate_until requests",
        )

        for req_idx, args in enumerate(iterator):
            if len(args) >= 3:
                context, gen_kwargs, aux_arguments = args
            elif len(args) == 2:
                context, gen_kwargs = args
                aux_arguments = None
            else:
                context = args[0]
                gen_kwargs = {}
                aux_arguments = None

            if not isinstance(gen_kwargs, dict):
                raise ValueError(
                    f"Expected generation kwargs as dict, got {type(gen_kwargs)}"
                )

            until = gen_kwargs.get("until")
            if until is not None and not isinstance(until, list):
                until = [until]

            image_tensors = self._images_to_tensors(aux_arguments)
            prompt = self._extract_prompt(context)
            if not prompt:
                raise ValueError("openvino-mx could not extract a text prompt.")
            if self.prompt_suffix:
                prompt = f"{prompt} {self.prompt_suffix}"

            if self.use_llamacpp:
                if image_tensors is not None:
                    raise ValueError(
                        "openvino-mx llama-cpp mode currently supports text-only requests and does not accept image input."
                    )
                thinking = gen_kwargs.get("enable_thinking")
                if thinking is not None:
                    thinking = self._as_bool(thinking)
                llama_kwargs = self._llamacpp_generate_kwargs(
                    gen_kwargs, self.default_max_new_tokens
                )
                generation_contract = self._generation_contract(
                    gen_kwargs, self.default_max_new_tokens, until
                )
                eval_logger.info(
                    "openvino-mx llama.cpp generation parameters: %s",
                    json.dumps(
                        {"enable_thinking": thinking, **llama_kwargs},
                        sort_keys=True,
                        default=str,
                    ),
                )
                text = self._llamacpp_generate(
                    prompt, until, gen_kwargs, enable_thinking=thinking
                )
                processed_text = self._apply_stop(text, until)
                if self.debug_token_log_path:
                    trace = self._last_llamacpp_trace
                    self._write_llamacpp_token_log(
                        self._extract_prompt(context),
                        trace["rendered_prompt"],
                        trace["input_token_ids"],
                        processed_text,
                        llama_kwargs,
                        generation_contract,
                        thinking,
                    )
            else:
                cfg = self.GenerationConfig()
                if self.rng_seed is not None and hasattr(cfg, "rng_seed"):
                    cfg.rng_seed = self.rng_seed
                cfg.max_new_tokens = int(
                    gen_kwargs.get("max_gen_toks", self.default_max_new_tokens)
                )
                for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
                    if key in gen_kwargs:
                        setattr(cfg, key, gen_kwargs[key])
                cfg.do_sample = self._as_bool(gen_kwargs.get("do_sample", gen_kwargs.get("temperature", 0) > 0))
                if until:
                    cfg.stop_strings = set(until)
                active_pipe = self.pipe if (image_tensors is not None or self.pipe is not None) else self.text_pipe
                thinking = gen_kwargs.get("enable_thinking", self.enable_thinking)
                if thinking is not None:
                    thinking = self._as_bool(thinking)
                prompt, use_gen_cfg_template = self._format_chat_prompt(
                    active_pipe, prompt, thinking
                )
                cfg.apply_chat_template = use_gen_cfg_template
                eval_logger.info(
                    "openvino-mx GenAI generation parameters: %s",
                    json.dumps(
                        {
                            "enable_thinking": thinking,
                            **{
                                key: getattr(cfg, key)
                                for key in (
                                    "max_new_tokens",
                                    "temperature",
                                    "top_p",
                                    "top_k",
                                    "min_p",
                                    "presence_penalty",
                                    "repetition_penalty",
                                    "do_sample",
                                )
                                if hasattr(cfg, key)
                            },
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )

                if image_tensors is None:
                    if not self.allow_text_only:
                        raise ValueError(
                            "openvino-mx requires image input in aux_arguments['visual']. "
                            "Set model_args allow_text_only=true to enable LLMPipeline fallback."
                        )
                    if self.pipe is not None:
                        # `pretrained` is a VLM-shaped export (multi-component
                        # IR with vision/embeddings/language sub-models, e.g.
                        # NNCF-quantized Qwen3.5 exports) -- VLMPipeline was
                        # already constructed successfully for it at __init__
                        # time. A text-only request (no images) must still go
                        # through THIS pipeline (VLMPipeline.generate accepts
                        # an empty images list for pure-text prompts), not
                        # self.text_pipe/LLMPipeline: LLMPipeline is built
                        # against the same VLM export but assumes a plain
                        # single-input text-generation graph signature, which
                        # a VLM export does not have -- it fails at actual
                        # inference time with "Port for tensor name input_ids
                        # was not found" (the VLM graph's real input port is
                        # named/shaped differently). This mirrors llm_bench's
                        # own --optimum/visual_text_gen path, which likewise
                        # always uses the VLM-aware pipeline for this model
                        # family even for text-only prompts, never a plain
                        # text pipeline.
                        try:
                            output = self.pipe.generate(prompt, [], cfg)
                        except RuntimeError as err:
                            raise RuntimeError(
                                f"openvino-mx text-only (VLMPipeline) generate_until failed at request index {req_idx}: {err}"
                            ) from err
                    else:
                        llm_kwargs = self._llm_generate_kwargs(
                            gen_kwargs, self.default_max_new_tokens
                        )
                        eval_logger.info(
                            "openvino-mx LLMPipeline generation parameters: %s",
                            json.dumps(
                                {"enable_thinking": thinking, **llm_kwargs},
                                sort_keys=True,
                                default=str,
                            ),
                        )
                        try:
                            if self.debug_token_log_path:
                                encoded, generated_ids, text, tokenizer = self._generate_token_trace(
                                    prompt, cfg
                                )
                                self._write_token_log(
                                    {
                                        "request_index": req_idx,
                                        "raw_prompt": self._extract_prompt(context),
                                        "rendered_prompt": prompt,
                                        "input_token_ids": [int(token_id) for token_id in encoded.input_ids.data[0]],
                                        "generated_token_ids": generated_ids,
                                        "generated_text": text,
                                        "generation_kwargs": llm_kwargs,
                                        "generation_contract": self._generation_contract(
                                            gen_kwargs, self.default_max_new_tokens, until
                                        ),
                                        "enable_thinking": thinking,
                                        "apply_chat_template": use_gen_cfg_template,
                                        "chat_template_path": self.chat_template_path,
                                        "tokenizer_source": "OpenVINOMX LLMPipeline tokenizer from GGUF",
                                        "tokenizer_details": {
                                            "implementation": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                                            "special_token_parsing": "handled by encode(add_special_tokens=True)",
                                            "add_special_tokens": True,
                                        },
                                    }
                                )
                                output = text
                            else:
                                output = self.text_pipe.generate(prompt, **llm_kwargs)
                        except RuntimeError as err:
                            raise RuntimeError(
                                f"openvino-mx text-only generate_until failed at request index {req_idx}: {err}"
                            ) from err
                else:
                    if self.pipe is None:
                        raise ValueError(
                            "openvino-mx received image input, but pretrained points to a "
                            "text-only model file that cannot initialize VLMPipeline."
                        )
                    try:
                        output = self.pipe.generate(prompt, image_tensors, cfg)
                    except RuntimeError as err:
                        image_shapes = [
                            tuple(getattr(tensor, "shape", ())) for tensor in image_tensors
                        ]
                        raise RuntimeError(
                            f"openvino-mx generate_until failed at request index {req_idx} with image shapes {image_shapes}: {err}"
                        ) from err
                text = self._extract_generated_text(output)
                processed_text = self._apply_stop(text, until)

            if self.debug_prompt_log_path and self._prompt_log_count < self.debug_prompt_log_max:
                try:
                    p_path = Path(self.debug_prompt_log_path)
                    p_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(p_path, "a", encoding="utf-8") as pf:
                        pf.write(
                            f"=== Sample {self._prompt_log_count + 1} (Request {req_idx}) ===\n"
                            f"--- [FULL PROMPT] ---\n{prompt}\n\n"
                            f"--- [FULL RESPONSE] ---\n{processed_text}\n\n"
                            + "=" * 60 + "\n\n"
                        )
                    self._prompt_log_count += 1
                except Exception as err:  # noqa: BLE001
                    eval_logger.warning("Failed to write debug_prompt_log_path: %s", err)

            results.append(processed_text)

        return results

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "openvino-mx currently supports generate_until only."
        )

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        raise NotImplementedError(
            "openvino-mx currently supports generate_until only."
        )
