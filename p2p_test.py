from multiprocessing.pool import ThreadPool
import os
from tqdm import tqdm 
import random
import json
from PIL import Image
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import sys
sys.path.append('/share/generation')
from diffusers import StableDiffusionPipeline, DiffusionPipeline, UniPCMultistepScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import argparse


MY_TOKEN = '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'
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# model_id = '/mnt/mnt/ve_share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5'
model_id = "/share/generation/models/online/diffusions/res/finetune/dreambooth/SD-HM-V0.5.0"
# ldm_stable = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=MY_TOKEN).to(device)
ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
ldm_stable.scheduler = UniPCMultistepScheduler.from_config(ldm_stable.scheduler.config)
tokenizer = ldm_stable.tokenizer


class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + \
            attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1,
                             1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1),
                              (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:],
                                             is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [
            item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(
            attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(
                    attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(
            prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(
            num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps,
                                               cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(
            prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * \
            self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps,
                                              cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(
            prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(
                attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * \
            self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps,
                                                cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(
                    len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, save_path='./res/cross_att.jpg'):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(
        attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), save_path=save_path)


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(
        attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(
        attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2),
                          3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))


def run_and_display(prompts, controller, save_path_1=None, save_path_2=None, latent=None, run_baseline=False, generator=None, split_save=True, save_path='./test.jpg'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator, save_path=save_path.replace('.jpg', '_wo.jpg'))
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                  guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    if split_save:
        Image.fromarray(images[0]).save(save_path_1)
        Image.fromarray(images[1]).save(save_path_2)
    else:
        # ptp_utils.view_images(images, save_path=save_path)
        pass
    return images, x_t

def replace_blend_reweight(prompts: list, words: tuple, latent_x, save_root: str, scene:str=None, id: int =0, cross_steps: float = 0.8, self_steps: float = 0.8, amplify_co: float = 2.0):
    word_blend = LocalBlend(prompts, words)
    controller_a = AttentionReplace(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=cross_steps, self_replace_steps=self_steps, local_blend=word_blend)

    equalizer = get_equalizer(prompts[1], (words[1],), (amplify_co,))
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=cross_steps, self_replace_steps=self_steps, equalizer=equalizer, local_blend=word_blend, controller=controller_a)
    save_func = "%s/replace_blend_reweight" % save_root
    if not os.path.exists(save_func):
        os.makedirs(save_func, exist_ok=True)
    save_id = "%s/%s_%d" % (save_func, scene, id)
    if not os.path.exists(save_id):
        os.makedirs(save_id, exist_ok=True)
    co_path = "%s/%.2f_%.2f_%.2f" % (save_id, cross_steps, self_steps, amplify_co)    
    save_path_1 = "%s/%s.png" % (co_path, "_".join(prompts[0].split()))
    save_path_2 = "%s/%s.png" % (co_path, "_".join(prompts[1].split()))
    if not os.path.exists(co_path):
        os.makedirs(co_path, exist_ok=True)
    _ = run_and_display(prompts, controller, latent=latent_x, run_baseline=False, save_path_1=save_path_1, save_path_2=save_path_2)
    return save_path_1, save_path_2
    

def refine(prompts: list, words: tuple, amplify_word: str, latent_x, save_root: str, scene:str=None, id: int =0, cross_steps: float = 0.8, self_steps: float = 0.8, amplify_co: float = 2.0):
    word_blend = LocalBlend(prompts, words)
    controller_a = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=cross_steps, self_replace_steps=self_steps, local_blend=word_blend)
    equalizer = get_equalizer(prompts[1], (amplify_word,), (2,))
    controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=cross_steps, self_replace_steps=self_steps, equalizer=equalizer, controller=controller_a, local_blend=word_blend)

    save_func = "%s/refine" % save_root
    if not os.path.exists(save_func):
        os.makedirs(save_func, exist_ok=True)
    save_id = "%s/%s_%d" % (save_func, scene, id)
    if not os.path.exists(save_id):
        os.makedirs(save_id, exist_ok=True)
    co_path = "%s/%.2f_%.2f_%.2f" % (save_id, cross_steps, self_steps, amplify_co)    
    save_path_1 = "%s/%s.png" % (co_path, "_".join(prompts[0].split()))
    save_path_2 = "%s/%s.png" % (co_path, "_".join(prompts[1].split()))
    if not os.path.exists(co_path):
        os.makedirs(co_path, exist_ok=True)
    _ = run_and_display(prompts, controller, latent=latent_x,  run_baseline=False, save_path_1=save_path_1, save_path_2=save_path_2)
    return save_path_1, save_path_2


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # Define command-line arguments
    parser.add_argument('--ORI_JSON_PATH', help='Specify a ori json path')
    parser.add_argument('--CO', type=float, help='Specify the coefficient')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the argument values
    ORI_JSON_PATH = args.ORI_JSON_PATH
    CO = args.CO

    # ORI_JSON_PATH = "/share/generation/data/train/diffusions/comb_cls/index_new.json"
    
    SAVE_ROOT = "/share/generation/data/p2p/imgs"
    from datetime import datetime
    current_time = datetime.now().time()
    NEW_JSON_PATH = "/share/generation/data/p2p/new_jsons/%s" % ORI_JSON_PATH.split("/")[-1]
    
    os.makedirs(SAVE_ROOT, exist_ok=True)

    g_cpu = torch.Generator().manual_seed(917)
    
    
    result = []
    with open(ORI_JSON_PATH, 'r') as file:
        data = json.load(file)
        # data = [_ for _ in data if _["scene"] in ["night"]]
        # data = random.sample(data, 10)
        
    for each in tqdm(data):
        prompts = [each["prompt_1"], each["prompt_2"]]
        id, scene = each["id"], each["scene"]

        # controller = AttentionStore()
        # image, x_t = run_and_display([prompts[0]], controller, latent=None, run_baseline=False, generator=g_cpu, split_save=False)

        words1 = prompts[0].split()
        words2 = prompts[1].split()
        different_words = []

        for word_1, word_2 in zip(words1, words2):
            if word_1 != word_2:
                different_words.append((word_1, word_2))

        if len(set(different_words)) == 1:
            save_path_1, save_path_2 = replace_blend_reweight(prompts, different_words[0], latent_x=None, save_root=SAVE_ROOT, scene=scene, id=id, cross_steps=CO, self_steps=CO)
        else:
            last_word = "" 
            for word_1, word_2 in zip(words1, words2):
                if word_1 != word_2:
                    break
                last_word = word_2
            save_path_1, save_path_2 = refine(prompts, words=(last_word, last_word), amplify_word="snow", latent_x=None, save_root=SAVE_ROOT, scene=scene, id=id, cross_steps=CO, self_steps=CO)
            
        each["img_path_1"] = save_path_1
        each["img_path_2"] = save_path_2
        
        result.append(each)
            
            
    with open(NEW_JSON_PATH, 'w') as json_file:
        json.dump(result, json_file, indent=4)
            
