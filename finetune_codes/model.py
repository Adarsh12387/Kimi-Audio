import os
import argparse
from typing import Optional, List
import shutil
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from .modeling_kimia import MoonshotKimiaForCausalLM


class KimiAudioModel(MoonshotKimiaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.whisper_model = WhisperEncoder("openai/whisper-large-v3", mel_batch_size=20, unfreeze_online_whisper_model=True)

    """@classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )
        state_dict = torch.load(cache_path)
        audio_model.load_state_dict(state_dict, strict=False)

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model"""
        
    
       
    '''@classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        device_map = "auto" if torch.cuda.is_available() else None

        if os.path.exists(model_name_or_path):
            cache_path = model_name_or_path
        else:
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            device_map=None,  # or "auto" in your modified version
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **model_load_kwargs,
        )


        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"),
            mel_batch_size=20,
            unfreeze_online_whisper_model=True
        )

        kimia_model = cls(audio_model.config)

        pretrained_state_dict = audio_model.state_dict()
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        valid_state_dict = {
            k: v for k, v in pretrained_state_dict.items()
            if k in kimia_model.state_dict() and kimia_model.state_dict()[k].shape == v.shape
        }

        kimia_model.load_state_dict(valid_state_dict, strict=False)

        if torch.cuda.is_available():
            kimia_model = kimia_model.to("cuda")

        return kimia_model'''
    
    '''@classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)
    
        # load base model
        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            **model_load_kwargs,
        )
        

        # load whisper encoder
        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), 
            mel_batch_size=20, 
            unfreeze_online_whisper_model=True
        )

        # initialize KimiAudioModel
        kimia_model = cls(audio_model.config)

        # merge audio model + whisper model state dicts
        pretrained_state_dict = audio_model.state_dict()
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        # --- Filter out invalid or mismatched tensors ---
        #print(pretrained_state_dict["model.layers.0.self_attn.q_proj.weight"].shape)
        #print(kimia_model.state_dict()["model.layers.0.self_attn.q_proj.weight"].shape)

        valid_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if v.numel() == 0:   # skip empty tensors
                #print(f"⚠️ Skipping {k}, shape={v.shape}")
                continue
            if k in kimia_model.state_dict() and kimia_model.state_dict()[k].shape == v.shape:
                valid_state_dict[k] = v
            else:
                if k in kimia_model.state_dict():
                    print(f"⚠️ Shape mismatch for {k}: {v.shape} vs {kimia_model.state_dict()[k].shape}")
                else:
                    print(f"⚠️ Key {k} not found in KimiAudioModel")

        # load filtered weights
        missing, unexpected = kimia_model.load_state_dict(valid_state_dict, strict=False)
        #print("✅ Loaded weights (with filtering)")
        #print("Missing keys:", missing)
        #print("Unexpected keys:", unexpected)

        return kimia_model'''
        
    @classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs=None, use_device_map=False, device_map="auto"):
        """
        Loads/prepares model safely.

        - model_load_kwargs: dict of kwargs to pass to from_pretrained (sanitized here)
        - use_device_map: if True, pass device_map=device_map to from_pretrained
                         (only valid if DeepSpeed is NOT using ZeRO-3)
        """

        import os, torch
        from transformers import AutoModelForCausalLM

        model_load_kwargs = dict(model_load_kwargs or {})

        # Canonicalize dtype keys: prefer 'dtype'
        if "torch_dtype" in model_load_kwargs and "dtype" not in model_load_kwargs:
            model_load_kwargs["dtype"] = model_load_kwargs.pop("torch_dtype")

        # Ensure low_cpu_mem_usage default only if not provided
        model_load_kwargs.setdefault("low_cpu_mem_usage", True)

        # Default dtype if none present
        if "dtype" not in model_load_kwargs:
            # Prefer bfloat16 if available, else float16
            try:
                model_load_kwargs["dtype"] = torch.bfloat16
            except Exception:
                model_load_kwargs["dtype"] = torch.float16

        # locate or download model
        if os.path.exists(model_name_or_path):
            cache_path = model_name_or_path
        else:
            from huggingface_hub import snapshot_download
            cache_path = snapshot_download(model_name_or_path)

        # Prevent accidental duplicate device_map in model_load_kwargs
        if "device_map" in model_load_kwargs:
            print("WARN: device_map found in model_load_kwargs; removing to avoid duplicates.")
            model_load_kwargs.pop("device_map")

        # Build final kwargs for from_pretrained
        final_kwargs = dict(model_load_kwargs)
        final_kwargs["trust_remote_code"] = final_kwargs.get("trust_remote_code", True)

        # Optionally pass device_map (only if you intentionally want it, and not using ZeRO-3)
        if use_device_map:
            final_kwargs["device_map"] = device_map

        # Call from_pretrained with cache_path as the first (positional) arg
        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            **final_kwargs
        )

        # (Optional) load whisper weights if present on disk (map to CPU)
        whisper_dir = os.path.join(cache_path, "whisper-large-v3")
        whisper_state = None
        if os.path.isdir(whisper_dir):
            whisper_model_bin = os.path.join(whisper_dir, "pytorch_model.bin")
            if os.path.isfile(whisper_model_bin):
                whisper_state = torch.load(whisper_model_bin, map_location="cpu")

        # Initialize your composite model on CPU
        kimia_model = cls(audio_model.config)

        # Merge state dicts (all on CPU)
        pretrained_state_dict = audio_model.state_dict()
        if whisper_state is not None:
            for n, p in whisper_state.items():
                pretrained_state_dict["whisper_model." + n] = p

        valid_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if getattr(v, "numel", lambda: 1)() == 0:
                continue
            if k in kimia_model.state_dict() and kimia_model.state_dict()[k].shape == v.shape:
                valid_state_dict[k] = v
            else:
                print(f"⚠️ Skipping {k}")

        kimia_model.load_state_dict(valid_state_dict, strict=False)

        print("✅ model initialized (use_device_map=%s). Note: if using DeepSpeed ZeRO-3, do NOT set use_device_map=True." % use_device_map)
        return kimia_model
        
    '''@classmethod
    def init_from_pretrained(cls, model_name_or_path, model_load_kwargs):
        if os.path.exists(model_name_or_path):
            # local path
            cache_path = model_name_or_path
        else:
            # cache everything if model_path is a model-id
            cache_path = snapshot_download(model_name_or_path)

        audio_model = AutoModelForCausalLM.from_pretrained(
            cache_path, 
            device_map=None,
            torch_dtype=torch.bfloat16, trust_remote_code=True, **model_load_kwargs,
        )

        whisper_model = WhisperEncoder(
            os.path.join(cache_path, "whisper-large-v3"), mel_batch_size=20, unfreeze_online_whisper_model=True
        )
        kimia_model = cls(audio_model.config)

        # merge audio model and whisper model's state dict
        pretrained_state_dict = audio_model.state_dict()
        
        for n, p in whisper_model.state_dict().items():
            pretrained_state_dict["whisper_model." + n] = p

        kimia_model.load_state_dict(pretrained_state_dict)

        return kimia_model'''

    
    '''@staticmethod
    def export_model(input_dir, output_dir):
        print("Loading model from {}".format(input_dir))
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        print("Saving Kimi-Audio LM to {}".format(output_dir))
        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {k: v for k, v in kimiaudio.state_dict().items() if not k.startswith("whisper_model")}
        audio_model.load_state_dict(audio_model_state_dict)

        audio_model.save_pretrained(output_dir)

        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py", os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py", os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperModel

        whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")

        kimiaudio_whisper_encoder_state_dict = {k.replace("speech_encoder.", "encoder."): v for k, v in kimiaudio.whisper_model.state_dict().items() if k.startswith("speech_encoder")}

        missing_keys, unexpected_keys = whisper_model.load_state_dict(kimiaudio_whisper_encoder_state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

        for k in missing_keys:
            assert k.startswith("decoder"), f"Missing keys: {k}"

        whisper_model.save_pretrained(os.path.join(output_dir, "whisper-large-v3"))

        print("Exported Kimi-Audio LM and Whisper model to {}".format(output_dir))'''
        
    def export_model(input_dir, output_dir, shared_whisper_dir):
        """
        Export Kimi-Audio LM (without whisper) and save a shared Whisper model once.

        Args:
            input_dir (str): Path to fine-tuned KimiAudio checkpoint.
            output_dir (str): Folder to save the exported LM.
            shared_whisper_dir (str): Parent folder to store a single shared Whisper model.
        """
        print(f"Loading Kimi-Audio model from {input_dir}")
        kimiaudio = KimiAudioModel.from_pretrained(input_dir)

        # -------------------
        # Export LM (without Whisper)
        # -------------------
        print(f"Saving Kimi-Audio LM to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        audio_model = MoonshotKimiaForCausalLM(kimiaudio.config)
        audio_model_state_dict = {
            k: v for k, v in kimiaudio.state_dict().items()
            if not k.startswith("whisper_model")
        }
        audio_model.load_state_dict(audio_model_state_dict)
        audio_model.save_pretrained(output_dir)

        # Copy config scripts
        shutil.copyfile("finetune_codes/configuration_moonshot_kimia.py",
                        os.path.join(output_dir, "configuration_moonshot_kimia.py"))
        shutil.copyfile("finetune_codes/modeling_kimia.py",
                        os.path.join(output_dir, "modeling_moonshot_kimia.py"))

        # -------------------
        # Export Whisper only once
        # -------------------
        if not os.path.exists(shared_whisper_dir):
            print(f"Creating shared Whisper model at {shared_whisper_dir}")
            os.makedirs(shared_whisper_dir, exist_ok=True)

            whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    
            # Map fine-tuned encoder weights
            encoder_state_dict = {
                k.replace("speech_encoder.", "encoder."): v
                for k, v in kimiaudio.whisper_model.state_dict().items()
                if k.startswith("speech_encoder")
            }

            missing_keys, unexpected_keys = whisper_model.load_state_dict(
                encoder_state_dict, strict=False
            )
            assert not unexpected_keys, f"Unexpected keys: {unexpected_keys}"

            # Save only once
            whisper_model.save_pretrained(shared_whisper_dir)
            print(f"Whisper model saved to {shared_whisper_dir}")
        else:
            print(f"Shared Whisper model already exists at {shared_whisper_dir}, skipping save.")

        print(f"✅ Export complete: Kimi-Audio LM -> {output_dir}, Whisper -> {shared_whisper_dir}")


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[torch.FloatTensor] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        whisper_input_feats = torch.from_numpy(whisper_input_feature[0]).unsqueeze(0)[:, :].to(torch.cuda.current_device())
        whisper_feats = self.whisper_model(whisper_input_feats)
        whisper_feats = whisper_feats.reshape(
            whisper_feats.shape[0],
            int(whisper_feats.shape[1] // 4),
            whisper_feats.shape[2] * 4,
        )
        return super().forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=return_dict,
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    parser.add_argument("--action", type=str, choices=["init_from_pretrained", "export_model"], default="init_from_pretrained")
    parser.add_argument("--output_dir", type=str, default="output/pretrained_hf")
    parser.add_argument("--input_dir", type=str, default="output/finetuned_hf")
    parser.add_argument("--whisper_dir", type=str, default="output/finetuned_hf")
    args = parser.parse_args()

    if args.action == "init_from_pretrained":

        model = KimiAudioModel.init_from_pretrained(args.model_name, model_load_kwargs={})

        os.makedirs(args.output_dir, exist_ok=True)
        # save model
        model.save_pretrained(args.output_dir)
    elif args.action == "export_model":
        KimiAudioModel.export_model(args.input_dir, args.output_dir,args.whisper_dir)
    else:
        raise ValueError(f"Invalid action: {args.action}")
