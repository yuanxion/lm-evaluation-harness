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
        self.use_llamacpp = bool(kwargs.get("llamacpp", False))
        self.llamacpp_chat = bool(kwargs.get("llamacpp_chat", False))
        self.llamacpp_n_ctx = kwargs.get("llamacpp_n_ctx")
        self.llamacpp_n_gpu_layers = kwargs.get("llamacpp_n_gpu_layers")
        self.llama = None

        if not find_spec("openvino"):
            raise ModuleNotFoundError(
                "package `openvino` is not installed. Please install it first."
            )

        if not find_spec("pipeline"):
            raise ModuleNotFoundError(
                "Could not import `pipeline`. Ensure your VLMPipeline module is importable."
            )

        import openvino as ov
        from pipeline import GenerationConfig, LLMPipeline, VLMPipeline

        self.ov = ov
        self.GenerationConfig = GenerationConfig
        pretrained_path = Path(self.pretrained)
        self.pipe = None
        if not pretrained_path.is_file():
            try:
                self.pipe = VLMPipeline(self.pretrained, self._device)
            except RuntimeError as err:
                # Not every OpenVINO IR directory is multimodal (e.g. a plain
                # optimum-cli text LLM export has no vision tower/language-model
                # split). Fall back to text-only via LLMPipeline below.
                eval_logger.warning(
                    "openvino-mx: VLMPipeline init failed for %s, falling back to "
                    "text-only LLMPipeline: %s",
                    self.pretrained,
                    err,
                )
                self.pipe = None
        self.allow_text_only = bool(kwargs.get("allow_text_only", True))
        self.prompt_suffix = str(kwargs.get("prompt_suffix", ""))
        self.text_pipe = None
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
            self.llama = Llama(self.pretrained, **llama_kwargs)
        else:
            self.text_pipe = LLMPipeline(self.pretrained, self._device)
        self.min_image_side = int(
            kwargs.get("min_image_side", self._infer_min_image_side(self.pretrained))
        )

        self.debug_image_dir = Path(kwargs.get("debug_image_dir", "openvino_mx_debug_images"))
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        self._image_dump_counter = 0

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
        if "repetition_penalty" in gen_kwargs:
            kwargs["repeat_penalty"] = float(gen_kwargs["repetition_penalty"])
        return kwargs

    def _llamacpp_generate(self, prompt: str, until: list[str] | None, gen_kwargs: dict) -> str:
        if self.llama is None:
            raise RuntimeError("openvino-mx llama-cpp mode is enabled but model is not initialized")

        llama_kwargs = self._llamacpp_generate_kwargs(gen_kwargs, self.default_max_new_tokens)
        if until:
            llama_kwargs["stop"] = until

        if self.llamacpp_chat:
            response = self.llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **llama_kwargs,
            )
            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return str(content)

        response = self.llama.create_completion(prompt=prompt, **llama_kwargs)
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

            if image_tensors is None:
                if not self.allow_text_only:
                    raise ValueError(
                        "openvino-mx requires image input in aux_arguments['visual']. "
                        "Set model_args allow_text_only=true to enable LLMPipeline fallback."
                    )
                try:
                    if self.use_llamacpp:
                        text = self._llamacpp_generate(prompt, until, gen_kwargs)
                        results.append(self._apply_stop(text, until))
                        continue

                    llm_kwargs = self._llm_generate_kwargs(
                        gen_kwargs, self.default_max_new_tokens
                    )
                    output = self.text_pipe.generate(prompt, **llm_kwargs)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"openvino-mx text-only generate_until failed at request index {req_idx}: {err}"
                    ) from err
            else:
                if self.use_llamacpp:
                    raise ValueError(
                        "openvino-mx llama-cpp mode currently supports text-only requests and does not accept image input."
                    )
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
            text = output.texts[0] if output and output.texts else ""
            results.append(self._apply_stop(text, until))

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