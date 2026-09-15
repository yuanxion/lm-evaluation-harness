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
        self.pipe = None
        if not pretrained_path.is_file():
            self.pipe = VLMPipeline(self.pretrained, self._device)
        self.allow_text_only = bool(kwargs.get("allow_text_only", True))
        self.prompt_suffix = str(kwargs.get("prompt_suffix", ""))

        self.ov_config = self._parse_ov_config(kwargs.get("ov_config"))

        # Parse apply_chat_template: default True unless specified False in kwargs or ov_config
        self.apply_chat_template = kwargs.get("apply_chat_template")
        if self.apply_chat_template is None and "apply_chat_template" in self.ov_config:
            self.apply_chat_template = bool(self.ov_config["apply_chat_template"])
        if self.apply_chat_template is None:
            self.apply_chat_template = True
        else:
            self.apply_chat_template = bool(self.apply_chat_template)

        # Parse enable_thinking: default False unless specified True in kwargs or ov_config
        self.enable_thinking = kwargs.get("enable_thinking")
        if self.enable_thinking is None and "enable_thinking" in self.ov_config:
            self.enable_thinking = bool(self.ov_config["enable_thinking"])
        if self.enable_thinking is None:
            self.enable_thinking = False
        else:
            self.enable_thinking = bool(self.enable_thinking)

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

    @staticmethod
    def _llm_generate_kwargs(gen_kwargs: dict, default_max_new_tokens: int) -> dict:
        kwargs = {
            "max_new_tokens": int(gen_kwargs.get("max_gen_toks", default_max_new_tokens))
        }
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if key in gen_kwargs:
                kwargs[key] = gen_kwargs[key]
        if "do_sample" in gen_kwargs:
            kwargs["do_sample"] = gen_kwargs["do_sample"]
        return kwargs

    def _format_chat_prompt(self, active_pipe, raw_prompt: str) -> tuple[str, bool]:
        if not self.apply_chat_template:
            return raw_prompt, False

        if self.enable_thinking is False:
            try:
                tokenizer = getattr(active_pipe, "get_tokenizer", lambda: None)()
                if tokenizer and hasattr(tokenizer, "apply_chat_template"):
                    formatted = tokenizer.apply_chat_template(
                        [{"role": "user", "content": raw_prompt}],
                        add_generation_prompt=True,
                        extra_context={"enable_thinking": False}
                    )
                    return formatted, False
            except Exception as err:  # noqa: BLE001
                eval_logger.warning("Failed to apply_chat_template with extra_context enable_thinking=False: %s", err)

        return raw_prompt, True

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

            cfg = self.GenerationConfig()
            cfg.max_new_tokens = int(
                gen_kwargs.get("max_gen_toks", self.default_max_new_tokens)
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

            active_pipe = self.pipe if (image_tensors is not None or self.pipe is not None) else self.text_pipe
            prompt, use_gen_cfg_template = self._format_chat_prompt(active_pipe, prompt)
            cfg.apply_chat_template = use_gen_cfg_template

            if image_tensors is None:
                if not self.allow_text_only:
                    raise ValueError(
                        "openvino-mx requires image input in aux_arguments['visual']. "
                        "Set model_args allow_text_only=true to enable LLMPipeline fallback."
                    )
                if self.pipe is not None:
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
                    try:
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
