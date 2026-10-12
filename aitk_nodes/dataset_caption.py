"""Captions the training dataset by asking the LLM node for every photo.

Runs on the training machine right after the dataset archive is extracted. Every
photo without a caption file is sent to the LLM node over HTTP and gets a
`trigger, caption` text file next to it. No model is loaded on the machine.
"""

import os
import re
import json
import base64

from server import PromptServer

from ..utils.fantasio_lib import load_fantasio_lib

CHAT_PATH = "/v1/chat/completions"
ANSWER_SCHEMA_NAME = "image_answer"
ANSWER_FIELD = "answer"
CAPTION_EXTENSION = ".txt"
PHOTO_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
FINISH_COMPLETE = "stop"
FINISH_TOKEN_LIMIT = "length"
UNIT_SENTENCE = "sentence"
UNIT_TAG = "tag"
UNIT_NONE = "none"
SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
TAG_END = re.compile(r",")
LONGEST_CUT_ESCAPE_LENGTH = 6
ERROR_BODY_LENGTH = 200


class LlmNodeUnreachable(Exception):
    pass


class LlmAnswerRejected(Exception):
    pass


def keep_complete_units(text, unit, limit):
    pattern = TAG_END if unit == UNIT_TAG else SENTENCE_END
    ends = list(pattern.finditer(text))

    if not ends:
        return ""

    last_kept = ends[min(limit, len(ends)) - 1] if limit else ends[-1]
    cut_at = last_kept.start() if unit == UNIT_TAG else last_kept.end()

    return text[:cut_at].strip()


def read_cut_json_string(content):
    opening = re.match(r'\s*\{\s*"' + ANSWER_FIELD + r'"\s*:\s*"', content)

    if not opening:
        return ""

    body = content[opening.end():]

    for dropped in range(min(LONGEST_CUT_ESCAPE_LENGTH, len(body)) + 1):
        try:
            return json.loads('"' + body[:len(body) - dropped] + '"')
        except ValueError:
            continue

    return ""


def read_answer(body, unit, limit):
    choices = body.get("choices") or []

    if not choices:
        raise LlmAnswerRejected("LLM node answered without choices")

    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    content = (choice.get("message") or {}).get("content") or ""

    if finish_reason == FINISH_TOKEN_LIMIT and unit != UNIT_NONE:
        answer = keep_complete_units(read_cut_json_string(content), unit, limit)
    elif finish_reason == FINISH_COMPLETE:
        answer = read_complete_answer(content)
    else:
        raise LlmAnswerRejected(f"LLM node stopped early: {finish_reason}")

    if not answer:
        raise LlmAnswerRejected("LLM node answered without a caption")

    return answer


def read_complete_answer(content):
    try:
        decoded = json.loads(content)
    except ValueError as error:
        raise LlmAnswerRejected("LLM node answered with content that is not JSON") from error

    if not isinstance(decoded, dict):
        raise LlmAnswerRejected("LLM node answered with content that is not a JSON object")

    return str(decoded.get(ANSWER_FIELD) or "").strip()


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
                "max_tokens": ("INT", {"default": 411, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "answer_unit": ([UNIT_SENTENCE, UNIT_TAG, UNIT_NONE],),
                "answer_unit_limit": ("INT", {"default": 0, "min": 0, "max": 1000}),
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

    def run(self, folder_path, llm_nodes_json, llm_api_key, instruction, trigger_word, max_tokens, temperature,
            top_p, answer_unit, answer_unit_limit, attempts, timeout_seconds, client_id=""):
        sid = client_id if client_id else None
        fantasio_lib = load_fantasio_lib()
        request = {
            "nodes": json.loads(llm_nodes_json),
            "api_key": llm_api_key,
            "instruction": instruction,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "unit": answer_unit,
            "limit": answer_unit_limit,
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
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": ANSWER_SCHEMA_NAME,
                    "schema": {
                        "type": "object",
                        "properties": {ANSWER_FIELD: {"type": "string"}},
                        "required": [ANSWER_FIELD],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
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

        return read_answer(body, request["unit"], request["limit"])

    def _photo_data_url(self, photo_path):
        mime_type = PHOTO_MIME_TYPES[os.path.splitext(photo_path)[1].lower()]

        with open(photo_path, "rb") as photo_file:
            encoded = base64.b64encode(photo_file.read()).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    def _emit(self, sid, message):
        PromptServer.instance.send_sync("progress", {"message": message}, sid)
