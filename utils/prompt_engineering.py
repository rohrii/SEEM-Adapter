import random
import numpy as np


def get_prompt_templates():
    prompt_templates = [
        '{}.',
        'a photo of a {}.',
        'a bad photo of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a photo of a hard to see {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a close-up photo of a {}.',
        'a pixelated photo of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a photo of the dirty {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a close-up photo of the {}.',
        'a low resolution photo of a {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a blurry photo of a {}.',
        'a pixelated photo of a {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'a photo of the large {}.',
        'a dark photo of a {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
    ]
    return prompt_templates

def prompt_engineering(classnames, topk=1, suffix='.'):
    prompt_templates = get_prompt_templates()
    temp_idx = np.random.randint(min(len(prompt_templates), topk))

    if isinstance(classnames, list):
        classname = random.choice(classnames)
    else:
        classname = classnames

    return prompt_templates[temp_idx].replace('.', suffix).format(classname.replace(',', '').replace('+', ' '))