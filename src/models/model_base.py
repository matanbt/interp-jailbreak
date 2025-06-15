from abc import ABC, abstractmethod
from typing import List
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from torch import Tensor
import torch
from jaxtyping import Float, Bool


class ModelBase(ABC):
    def __init__(self, model_name_or_path: str, 
        model_obj: PreTrainedModel = None,  # for testing
    ):
        self.model_name_or_path = model_name_or_path.lower()
        self.model: PreTrainedModel = model_obj or self._load_model(model_name_or_path)
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer(model_name_or_path)
        self.device = self.model.device
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.refusal_toks = self._get_refusal_toks()
        self.affirm_toks = self._get_affirm_toks()

        self.before_instr_tok_count, self.after_instr_tok_count = self._get_before_after_instr_tok_count() 

        self.short_name = self.model_name_or_path.split('/')[-1]
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

        self.tl_model = None

        self._post_init_validations()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    def _post_init_validations(self):
        if self.model.dtype not in [torch.float16, torch.bfloat16]:
            print("[WARNING]", f"Model dtype is (probably) too big: {self.model.dtype}")
        assert self.tokenizer.chat_template is not None, "Tokenizer does not have a chat template"

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> PreTrainedModel:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass
   
    def generate_batch(
            self,
            messages: List[str],
            prefix_fillers: List[str] = None,  # optional strings to force the model to start with (per message)
            return_full_chat=False,
            wo_tempalte_chat_suffix=False,
            batch_size=8, max_new_tokens=256) -> List[str]:
        responses = []
        for i in tqdm(range(0, len(messages), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(
                instructions=messages[i:i + batch_size],
                outputs=prefix_fillers[i:i + batch_size] if prefix_fillers else None,
                wo_tempalte_chat_suffix=wo_tempalte_chat_suffix,
                ).to(self.model.device)

            generation_toks = self.model.generate(
                **tokenized_instructions,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            if not return_full_chat:
                tokenized_without_outputs = self.tokenize_instructions_fn(instructions=messages[i:i + batch_size]).to(self.model.device)
                generation_toks = generation_toks[:, tokenized_without_outputs.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                responses.append(self.tokenizer.decode(generation, skip_special_tokens=not return_full_chat).strip())

        return responses

    def get_activations(self, messages, force_output_prefixes=None):
        # TODO support multiple messages
        assert len(messages) == 1 and isinstance(messages, list), "Only one message is supported for now"

        inputs = self.tokenize_instructions_fn(instructions=messages, outputs=force_output_prefixes).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        
        hs = outputs['hidden_states']
        hs = torch.stack([h[0, ...] for h in hs], dim=0)
        hs = hs.detach().cpu()
        
        attns = outputs['attentions']
        attns = torch.stack([attn[0, ...] for attn in attns], dim=0)
        attns = attns.detach().cpu()

        logits = outputs['logits'][0, -1]  # last token
        logits = logits.detach().cpu()
        
        return (hs,  # n_layers, seq_len, hidden_dim
                attns, # n_layers, n_heads, seq_len, seq_len
                logits) # vocab_size
    
    def calc_gcg_ce_loss(self, 
                        messages: List[str], 
                        target: str) -> List[float]:
        """Calculate CE loss for target as a response-prefix for the given messages."""
        assert isinstance(messages, list) and isinstance(target, str), "expects list of messages and a target string"

        inputs = self.tokenize_instructions_fn(instructions=messages, outputs=[target]).to(self.model.device)
        targets = self.tokenizer([target], padding=False, add_special_tokens=False, return_tensors="pt").to(self.device)

        logits = self.model(**inputs).logits

        inputs, targets = inputs['input_ids'], targets['input_ids']
        tmp = inputs.shape[1] - targets.shape[1]
        shift_logits = logits[..., tmp-1:-1, :].contiguous()
        shift_labels = targets.repeat(len(messages), 1)

        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
        loss = loss.view(len(messages), -1).mean(dim=-1).tolist()
        
        return loss

    def calc_sim_with_dir(
            self,
            messages: List[str],
            direction: Float[Tensor, "d_model"]=None,  # defaults to the refusal direction
            layer: int = 15,
            tok_pos: int = -1,
    ):
        inputs = self.tokenize_instructions_fn(instructions=messages).to(self.device)

        hidden_states = self.model(**inputs, output_hidden_states=True)['hidden_states']

        direction = direction or self.refusal_dir

        return torch.nn.functional.cosine_similarity(
            hidden_states[layer][:, tok_pos, :],
            direction,
            dim=-1
        ).tolist()  # TODO take mean per message!

    def to_toks(self, instruction, add_template=True, output=None):  # TODO use a more general function instead
        if not add_template:
            return self.tokenizer.encode(instruction, return_tensors="pt", add_special_tokens=False)
        if output is not None:
            return self.tokenize_instructions_fn(instructions=[instruction], outputs=[output]).input_ids
        return self.tokenize_instructions_fn(instructions=[instruction]).input_ids
