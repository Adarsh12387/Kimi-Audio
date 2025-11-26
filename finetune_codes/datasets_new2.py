from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
import librosa

from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent


class LazySupervisedDataset(Dataset):
    """
    Dataset for supervised fine-tuning (Speech-to-Speech or Speech-to-Text).
    Autoregressive version: shifts labels by one token for next-token prediction.

    Each data entry:
    {
        "task_type": "understanding" | "speech_to_speech",
        "conversation": [
            {
                "role": "user" | "assistant",
                "message_type": "text" | "audio" | None,
                "content": str,  # text or audio file path
                "audio_tokens": list[int],  # precomputed tokens (optional)
                "whisper_feat": optional list or tensor
            }
        ]
    }
    """

    def __init__(self, raw_data_list: List[Dict], whisper_model, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__()
        self.raw_data = raw_data_list
        self.whisper_model = whisper_model
        self.text_tokenizer = text_tokenizer
        self.max_len = max_len
        self.kimia_token_offset = kimia_token_offset

        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.pad_token = self.extra_tokens.pad

        print(f"[LazySupervisedDataset] Loaded {len(raw_data_list)} samples.")

    def __len__(self):
        return len(self.raw_data)

    # ----------------------------------------------------------------------
    # 1. Feature extraction (Whisper)
    # ----------------------------------------------------------------------
    def extract_whisper_feat(self, wav_path: str):
        """Extract continuous embeddings from Whisper model."""
        wav, sr = librosa.load(wav_path, sr=16000)
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(torch.cuda.current_device())
        with torch.no_grad():
            features = self.whisper_model(wav_tensor)
        # reshape if model downsamples
        features = features.reshape(features.shape[0], features.shape[1] // 4, features.shape[2] * 4)
        return features.cpu()

    # ----------------------------------------------------------------------
    # 2. Text tokenization helper
    # ----------------------------------------------------------------------
    def _tokenize_text(self, text: str):
        return [] if text is None else self.text_tokenizer.encode(text, bos=False, eos=False)

    # ----------------------------------------------------------------------
    # 3. Tokenize a single message
    # ----------------------------------------------------------------------
    def tokenize_message(
        self,
        message: Dict,
        tokenize_role: bool = True,
        add_ct_token: bool = False,
        add_msg_end_token: bool = False,
        extract_whisper_feature: bool = False
    ) -> KimiAContent:
        """Tokenize one message into KimiAContent."""
        kimia_msg = KimiAContent()
        role = message.get("role")
        msg_type = message.get("message_type")
        content = message.get("content")
        is_target = role == "assistant"

        # Role control tokens
        if tokenize_role:
            if role == "user":
                kimia_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                kimia_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise ValueError(f"Invalid role: {role}")

        # Text message
        if msg_type == "text" and content:
            text_tokens = self._tokenize_text(content)
            kimia_msg.text_extend(text_tokens, text_token_loss_mask=is_target)
            kimia_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

            if is_target:
                kimia_msg.text_append(self.extra_tokens.kimia_text_eos, text_token_loss_mask=True)
                kimia_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)


        # Audio message
        elif msg_type == "audio" and content:
            audio_tokens = message.get("audio_tokens", [])
            kimia_msg.audio_append(self.extra_tokens.media_begin)
            kimia_msg.audio_extend(audio_tokens, is_continuous=True, audio_token_loss_mask=is_target)
            kimia_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=is_target)
            kimia_msg.text_extend([self.extra_tokens.kimia_text_blank] * (len(audio_tokens) + 2))

            if extract_whisper_feature:
                whisper_feat = (
                    torch.tensor(message["whisper_feat"])
                    if "whisper_feat" in message
                    else self.extract_whisper_feat(content)
                )
                kimia_msg.continuous_feature.append(whisper_feat)

        elif msg_type is None:
            pass
        else:
            raise ValueError(f"Unknown message_type: {msg_type}")

        # Add optional CT or END tokens
        if add_ct_token:
            kimia_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)
        if add_msg_end_token:
            kimia_msg.audio_append(self.extra_tokens.msg_end)
            kimia_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert kimia_msg.is_valid(), f"Invalid KimiAContent: {kimia_msg}"
        return kimia_msg

    # ----------------------------------------------------------------------
    # 4. Tokenize full conversation
    # ----------------------------------------------------------------------
    def tokenize_conversation(self, messages: List[Dict], add_assistant_start_msg: bool = True) -> KimiAContent:
        msgs = []
        previous_role = None

        for i, message in enumerate(messages):
            tokenize_role = previous_role != message["role"]
            add_ct_token = (i == len(messages) - 1) or (messages[i + 1]["role"] != message["role"])
            add_msg_end_token = add_ct_token
            previous_role = message["role"]

            msg_obj = self.tokenize_message(
                message,
                tokenize_role=tokenize_role,
                add_ct_token=add_ct_token,
                add_msg_end_token=add_msg_end_token,
                extract_whisper_feature=True,
            )
            msgs.append(msg_obj)

        if add_assistant_start_msg:
            assistant_start = self.tokenize_message(
                {"role": "assistant", "message_type": None},
                tokenize_role=True,
            )
            msgs.append(assistant_start)

        merged = msgs[0]
        for m in msgs[1:]:
            merged.merge(m)
        return merged

    # ----------------------------------------------------------------------
    # 5. Main dataset output
    # ----------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one sample for autoregressive training."""
        entry = self.raw_data[idx]
        tokenized_conv = self.tokenize_conversation(entry["conversation"], add_assistant_start_msg=False)

        # Convert to tensors
        audio_input_ids, text_input_ids, is_continuous_mask, audio_token_loss_mask, text_token_loss_mask = tokenized_conv.to_tensor()
        audio_features = tokenized_conv.continuous_feature

        # ===== Autoregressive label shifting =====
        def shift_right(inputs, pad_token):
            return torch.cat((inputs[:, 1:], torch.full_like(inputs[:, :1], pad_token)), dim=1)

        def shift_mask(mask):
            return torch.cat((mask[:, 1:], torch.zeros_like(mask[:, :1], dtype=torch.bool)), dim=1)

        audio_labels = shift_right(audio_input_ids, self.pad_token)
        text_labels = shift_right(text_input_ids, self.pad_token)
        audio_loss_mask = shift_mask(audio_token_loss_mask)
        text_loss_mask = shift_mask(text_token_loss_mask)

        # Mask padding regions
        audio_labels = audio_labels.masked_fill(~audio_loss_mask, self.pad_token)
        text_labels = text_labels.masked_fill(~text_loss_mask, self.pad_token)

        return {
            "input_ids": audio_input_ids,
            "text_input_ids": text_input_ids,
            "whisper_input_feature": audio_features,
            "is_continuous_mask": is_continuous_mask,
            "labels": (audio_labels, text_labels),
        }

    # ----------------------------------------------------------------------
    # 6. Collate (micro-batch = 1)
    # ----------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1
        return batch[0]

