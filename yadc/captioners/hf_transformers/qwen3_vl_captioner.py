from transformers import AutoConfig, Qwen3VLMoeForConditionalGeneration

from yadc.core import logging

from .base import BaseTransformersCaptioner

QWEN_REPOS = [
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
]

QWEN_MODULE_COMPILATION = {
    'Gemma3nModel': {
        'mode': 'default',
        'dynamic': True,
        'fullgraph': True,
    },
}

_logger = logging.get_logger(__name__)

class Qwen3VLCaptioner(BaseTransformersCaptioner):
    def __init__(self, **kwargs):
        kwargs['_model_name'] = 'Qwen3 VL'
        kwargs['_model_repo_list'] = QWEN_REPOS
        kwargs['_module_compilation'] = QWEN_MODULE_COMPILATION

        kwargs['_torch_compile'] = False

        super().__init__(**kwargs)

    def _from_pretrained(self, model_repo: str, **kwargs):
        from huggingface_hub import snapshot_download
        from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

        checkpoint_path = snapshot_download(
            model_repo,
            local_files_only=kwargs.get('local_files_only', False),
        )

        config = AutoConfig.from_pretrained(model_repo)

        with init_empty_weights():
            model = Qwen3VLMoeForConditionalGeneration(config)

        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=['Qwen3VLMoeVisionModel', 'Qwen3VLMoeTextDecoderLayer'],
        )

        for name in list(device_map.keys()):
            if name.startswith('lm_head'):
                device_map[name] = 0
            elif name.startswith('model.visual'):
                device_map[name] = 0
            elif name.startswith('model.language_model.'):
                if '.gate' in name:
                    device_map[name] = 0
                elif '.layer' in name:
                    device_map[name] = 'cpu'
            elif device_map[name] == 'disk':
                device_map[name] = 'cpu'

        print(device_map)

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            # offload_folder='.cache',
            device_map=device_map,
            no_split_module_classes=['Qwen3VLMoeVisionModel', 'Qwen3VLMoeTextDecoderLayer'],
        )

        return model
