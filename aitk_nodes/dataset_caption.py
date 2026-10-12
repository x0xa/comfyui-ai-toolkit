"""Captions the training dataset by asking the LLM node for every photo.

Runs on the training machine right after the dataset archive is extracted. Every
photo without a caption file is sent to the LLM node over HTTP and gets a
`trigger, caption` text file next to it. No model is loaded on the machine.

The answer is bound to a pattern (`answer_pattern`, a regular expression the LLM
node enforces while it writes): the node ends the answer itself as soon as the
pattern is complete. An answer that still hit the token limit is an error - its
text is never used.
"""

import os
import json
import base64

from server import PromptServer

from ..utils.fantasio_lib import load_fantasio_lib

CHAT_PATH = "/v1/chat/completions"
CAPTION_EXTENSION = ".txt"
PHOTO_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
FINISH_COMPLETE = "stop"
ERROR_BODY_LENGTH = 200


class LlmNodeUnreachable(Exception):
    pass


class LlmAnswerRejected(Exception):
    pass


def read_answer(body):
    choices = body.get("choices") or []

    if not choices:
        raise LlmAnswerRejected("LLM node answered without choices")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")

    if finish_reason != FINISH_COMPLETE:
        raise LlmAnswerRejected(f"LLM node stopped early: {finish_reason}")

    answer = ((choice.get("message") or {}).get("content") or "").strip()

    if not answer:
        raise LlmAnswerRejected("LLM node answered without a caption")

    return answer


class FantasioCaptionDataset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"multiline": False}),
                "llm_nodes_json": ("STRING", {"multiline": False}),
                "llm_api_key": ("STRING", {"default": ""}),
                "instruction": ("STRING", {"multiline": True}),
                "trigger_word": ("STRING", {"multiline": False}),
                "answer_pattern": ("STRING", {"multiline": False}),
                "max_tokens": ("INT", {"default": 400, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "attempts": ("INT", {"default": 3, "min": 1, "max": 10}),
                "timeout_seconds": ("INT", {"default": 45, "min": 5, "max": 600}),
            },
            "hidden": {
                "client_id": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder_path",)
    FUNCTION = "run"
    CATEGORY = "AI Toolkit/Fantasio"

    def run(self, folder_path, llm_nodes_json, llm_api_key, instruction, trigger_word, answer_pattern, max_tokens,
            temperature, top_p, attempts, timeout_seconds, client_id=""):
        sid = client_id if client_id else None
        fantasio_lib = load_fantasio_lib()
        request = {
            "nodes": json.loads(llm_nodes_json),
            "api_key": llm_api_key,
            "instruction": instruction,
            "answer_pattern": answer_pattern,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "attempts": attempts,
            "timeout_seconds": timeout_seconds,
        }

        try:
            photos = self._photos_without_caption(folder_path)

            for index, photo_name in enumerate(photos):
                self._emit(sid, f"Describing photo {index + 1} of {len(photos)}")
                caption = self._describe(fantasio_lib, request, os.path.join(folder_path, photo_name))
                self._write_caption(folder_path, photo_name, f"{trigger_word}, {caption}")

            self._emit(sid, f"Captions ready: {len(photos)} described")
            return (folder_path,)
        except Exception as e:
            self._emit(sid, f"Captioning failed: {e}")
            raise

    def _photos_without_caption(self, folder_path):
        names = sorted(
            name for name in os.listdir(folder_path)
            if os.path.splitext(name)[1].lower() in PHOTO_MIME_TYPES
        )

        if not names:
            raise RuntimeError(f"Dataset folder {folder_path} holds no photos")

        return [name for name in names if not self._has_caption(folder_path, name)]

    def _has_caption(self, folder_path, photo_name):
        caption_path = self._caption_path(folder_path, photo_name)

        if not os.path.isfile(caption_path):
            return False

        with open(caption_path, "r", encoding="utf-8") as caption_file:
            return caption_file.read().strip() != ""

    def _caption_path(self, folder_path, photo_name):
        return os.path.join(folder_path, os.path.splitext(photo_name)[0] + CAPTION_EXTENSION)

    def _write_caption(self, folder_path, photo_name, caption):
        with open(self._caption_path(folder_path, photo_name), "w", encoding="utf-8") as caption_file:
            caption_file.write(caption)

    def _describe(self, fantasio_lib, request, photo_path):
        nodes = request["nodes"]
        tried_urls = set()
        last_failure = None

        for _ in range(request["attempts"]):
            node = nodes[0]

            try:
                return self._ask(fantasio_lib, request, node, photo_path)
            except LlmAnswerRejected as error:
                last_failure = error
            except LlmNodeUnreachable as error:
                last_failure = error
                tried_urls.add(node["url"])
                untried = [candidate for candidate in nodes if candidate["url"] not in tried_urls]

                if untried:
                    nodes.remove(untried[0])
                    nodes.insert(0, untried[0])

        raise RuntimeError(
            f"Photo {os.path.basename(photo_path)} was not described in {request['attempts']} attempts: {last_failure}"
        )

    def _ask(self, fantasio_lib, request, node, photo_path):
        payload = {
            "model": node["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request["instruction"]},
                        {"type": "image_url", "image_url": {"url": self._photo_data_url(photo_path)}},
                    ],
                },
            ],
            "max_tokens": request["max_tokens"],
            "temperature": request["temperature"],
            "top_p": request["top_p"],
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"regex": request["answer_pattern"]},
        }

        try:
            status, raw_body = fantasio_lib.post_json(
                node["url"].rstrip("/") + CHAT_PATH,
                payload,
                headers={"Authorization": f"Bearer {request['api_key']}"},
                timeout_seconds=request["timeout_seconds"],
            )
        except fantasio_lib.InstanceEgressUnavailable:
            raise
        except RuntimeError as error:
            raise LlmNodeUnreachable(str(error)) from error

        if status != 200:
            body_start = raw_body[:ERROR_BODY_LENGTH].decode("utf-8", errors="replace")
            raise LlmNodeUnreachable(f"LLM node answered {status}: {body_start}")

        try:
            body = json.loads(raw_body)
        except ValueError as error:
            raise LlmNodeUnreachable("LLM node answered with a body that is not JSON") from error

        return read_answer(body)

    def _photo_data_url(self, photo_path):
        mime_type = PHOTO_MIME_TYPES[os.path.splitext(photo_path)[1].lower()]

        with open(photo_path, "rb") as photo_file:
            encoded = base64.b64encode(photo_file.read()).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    def _emit(self, sid, message):
        PromptServer.instance.send_sync("progress", {"message": message}, sid)
