"""
Model inference utilities for extracting hidden states and generating responses.

Supports:
- QwQ (reasoning model)
- DeepSeek-R1-Distill (reasoning model)
- Qwen2.5-Instruct (instruction-tuned baseline)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class ModelWrapper:
    """Wrapper for model inference with hidden state extraction."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False
    ):
        """
        Initialize model wrapper.

        Args:
            model_name: HuggingFace model name
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.device = device

        print(f"Loading {model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            load_in_8bit=load_in_8bit,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if not load_in_8bit and device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()

        print(f"Model loaded on {device}")
        print(f"Number of layers: {self.model.config.num_hidden_layers}")

    @torch.no_grad()
    def generate_with_hidden_states(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        extract_layers: Optional[List[int]] = None,
        extract_position: str = "last"  # "last", "all", or specific index
    ) -> Dict:
        """
        Generate text and extract hidden states.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            extract_layers: List of layer indices to extract (None = all layers)
            extract_position: Which token position(s) to extract

        Returns:
            Dictionary containing generated text and hidden states
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        input_length = inputs.input_ids.shape[1]

        # Generate with output_hidden_states only if needed
        need_hidden_states = extract_layers is not None
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_hidden_states=need_hidden_states,
            return_dict_in_generate=True,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode generated text
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][input_length:],
            skip_special_tokens=True
        )

        # Extract hidden states
        # Note: hidden_states is a tuple of tuples: (layers,) for each generation step
        # Each layer tuple contains tensors of shape (batch, seq_len, hidden_dim)
        hidden_states_per_layer = []

        if extract_layers is None:
            extract_layers = list(range(self.model.config.num_hidden_layers))

        # For simplicity, extract from the last generation step
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_step_hidden = outputs.hidden_states[-1]  # Last generation step

            for layer_idx in extract_layers:
                layer_hidden = last_step_hidden[layer_idx]  # Shape: (1, seq_len, hidden_dim)

                if extract_position == "last":
                    # Extract last token's hidden state
                    hidden = layer_hidden[0, -1, :].cpu().numpy()
                elif extract_position == "all":
                    # Extract all tokens' hidden states
                    hidden = layer_hidden[0, :, :].cpu().numpy()
                else:
                    # Extract specific position
                    pos = int(extract_position)
                    hidden = layer_hidden[0, pos, :].cpu().numpy()

                hidden_states_per_layer.append(hidden)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": prompt + generated_text,
            "hidden_states": hidden_states_per_layer,
            "layers_extracted": extract_layers,
            "input_length": input_length
        }

    @torch.no_grad()
    def get_hidden_states_for_text(
        self,
        text: str,
        extract_layers: Optional[List[int]] = None,
        extract_position: str = "last"
    ) -> np.ndarray:
        """
        Extract hidden states for given text (forward pass only, no generation).

        Args:
            text: Input text
            extract_layers: List of layer indices to extract
            extract_position: Which token position to extract

        Returns:
            Array of hidden states
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True
        )

        if extract_layers is None:
            extract_layers = list(range(self.model.config.num_hidden_layers))

        hidden_states_list = []

        for layer_idx in extract_layers:
            layer_hidden = outputs.hidden_states[layer_idx]

            if extract_position == "last":
                hidden = layer_hidden[0, -1, :].cpu().numpy()
            elif extract_position == "all":
                hidden = layer_hidden[0, :, :].cpu().numpy()
            else:
                pos = int(extract_position)
                hidden = layer_hidden[0, pos, :].cpu().numpy()

            hidden_states_list.append(hidden)

        return np.array(hidden_states_list)


def batch_inference(
    model_wrapper: ModelWrapper,
    prompts: List[str],
    max_new_tokens: int = 512,
    extract_layers: Optional[List[int]] = None,
    show_progress: bool = True
) -> List[Dict]:
    """
    Run inference on a batch of prompts.

    Args:
        model_wrapper: ModelWrapper instance
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        extract_layers: Layers to extract hidden states from
        show_progress: Show progress bar

    Returns:
        List of results
    """
    results = []

    iterator = tqdm(prompts) if show_progress else prompts

    for prompt in iterator:
        result = model_wrapper.generate_with_hidden_states(
            prompt,
            max_new_tokens=max_new_tokens,
            extract_layers=extract_layers
        )
        results.append(result)

    return results


if __name__ == "__main__":
    # Test model loading
    print("Testing model wrapper...")

    # You can test with a smaller model first
    test_model = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing

    wrapper = ModelWrapper(test_model)

    test_prompt = "What is 2 + 2?"

    result = wrapper.generate_with_hidden_states(
        test_prompt,
        max_new_tokens=50,
        extract_layers=[0, -1]  # First and last layer
    )

    print("\nGenerated text:")
    print(result["generated_text"])
    print(f"\nExtracted hidden states from {len(result['hidden_states'])} layers")
    print(f"Hidden state shape: {result['hidden_states'][0].shape}")
