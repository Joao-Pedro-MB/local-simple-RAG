"""Image describer: describes images using a local vision model.

The production adapter uses Gemma 4 E2B's vision projector through llama-cpp-python
at temperature 0.
"""

from pathlib import Path


class GemmaImageDescriber:
    """Gemma 4 E2B with its vision projector, run in-process on the CPU by llama.cpp.

    The model and projector weights live in the Hugging Face cache, outside the repository.
    """

    REPO = "google/gemma-4-E2B-it-qat-q4_0-gguf"
    MODEL_FILE = "gemma-4-E2B_q4_0-it.gguf"
    MMPROJ_FILE = "gemma-4-E2B-it-mmproj.gguf"
    REVISION = "675cff42a74c774d6cb76f76d8eacb49b48c9b93"

    DEFAULT_PROMPT = "Describe this image in detail."

    def __init__(self, context_window=4096, max_tokens=256, prompt=DEFAULT_PROMPT):
        import llama_cpp
        from huggingface_hub import hf_hub_download
        from llama_cpp.llama_chat_format import Gemma4ChatHandler

        self.context_window = context_window
        self.max_tokens = max_tokens
        self.prompt = prompt

        model_path = hf_hub_download(self.REPO, self.MODEL_FILE, revision=self.REVISION)
        mmproj_path = hf_hub_download(self.REPO, self.MMPROJ_FILE, revision=self.REVISION)

        chat_handler = Gemma4ChatHandler(clip_model_path=mmproj_path, verbose=False, use_gpu=False)
        self._llm = llama_cpp.Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_ctx=context_window,
            verbose=False,
        )

    def describe(self, image_path: Path | str) -> str:
        """Describe an image file using Gemma's vision projector at temperature 0."""
        import base64

        path = Path(image_path)
        data = path.read_bytes()
        ext = path.suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        elif not ext:
            ext = "png"
        b64 = base64.b64encode(data).decode("utf-8")
        data_uri = f"data:image/{ext};base64,{b64}"

        response = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        content = response["choices"][0]["message"]["content"] or ""
        return content.strip()
