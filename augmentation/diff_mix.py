import os
from typing import Callable, Tuple

import torch
from PIL import Image
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
import glob

from augmentation.base_augmentation import GenerativeMixup
from diffusers import (
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.utils import logging

os.environ["WANDB_DISABLED"] = "true"
ERROR_MESSAGE = "Tokenizer already contains the token {token}. \
Please pass a different `token` that is not already in the tokenizer."


def format_name(name):
    return f"<{name.replace(' ', '_')}>"


def load_diffmix_embeddings(
    embed_path: str,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device="cuda",
):

    embedding_ckpt = torch.load(embed_path, map_location="cpu")
    learned_embeds_dict = embedding_ckpt["learned_embeds_dict"]
    name2placeholder = embedding_ckpt["name2placeholder"]
    placeholder2name = embedding_ckpt["placeholder2name"]

    name2placeholder = {
        k.replace("/", " ").replace("_", " "): v for k, v in name2placeholder.items()
    }
    placeholder2name = {
        v: k.replace("/", " ").replace("_", " ") for k, v in name2placeholder.items()
    }

    for token, token_embedding in learned_embeds_dict.items():

        # add the token in tokenizer
        num_added_tokens = tokenizer.add_tokens(token)
        assert num_added_tokens > 0, ERROR_MESSAGE.format(token=token)

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))
        added_token_id = tokenizer.convert_tokens_to_ids(token)

        # get the old word embeddings
        embeddings = text_encoder.get_input_embeddings()

        # get the id for the token and assign new embeds
        embeddings.weight.data[added_token_id] = token_embedding.to(
            embeddings.weight.dtype
        )

    return name2placeholder, placeholder2name


def identity(*args):
    return args


class IdentityMap:
    def __getitem__(self, key):
        return key


class DDIMLoraMixup(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        lora_path: str,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        embed_path: str = None,
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        disable_safety_checker: bool = True,
        revision: str = None,
        device="cuda",
        ddim_eta: float = 0.0,
        num_inference_steps: int = 50,
        **kwargs,
    ):

        super(DDIMLoraMixup, self).__init__()

        if DDIMLoraMixup.pipe is None:

            PipelineClass = StableDiffusionImg2ImgPipeline

            DDIMLoraMixup.pipe = PipelineClass.from_pretrained(
                    model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                    torch_dtype=torch.float16,
                ).to(device)

            scheduler = DDIMScheduler.from_config(
                DDIMLoraMixup.pipe.scheduler.config, local_files_only=True
            )
            self.placeholder2name = {}
            self.name2placeholder = {}
            if embed_path is not None:
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(
                    embed_path,
                    DDIMLoraMixup.pipe.text_encoder,
                    DDIMLoraMixup.pipe.tokenizer,
                )
            if lora_path is not None:
                DDIMLoraMixup.pipe.load_lora_weights(lora_path)
            DDIMLoraMixup.pipe.scheduler = scheduler

            print(f"successfuly load lora weights from {lora_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name
        self.ddim_eta = ddim_eta
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        image: Image.Image,
        label: int,
        metadata: dict,
        strength: float = 0.5,
        resolution=512,
    ) -> Tuple[Image.Image, int]:

        canvas = [img.resize((resolution, resolution), Image.BILINEAR) for img in image]
        name = metadata.get("name", "")

        if self.name2placeholder is not None:
            name = self.name2placeholder[name]
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        print(prompt)

        kwargs = dict(
            image=canvas,
            prompt=[prompt],
            strength=strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=len(canvas),
            eta=self.ddim_eta,  # DDIM parameter
        )

        has_nsfw_concept = True
        while has_nsfw_concept:
            with autocast("cuda"):
                outputs = self.pipe(**kwargs)

            has_nsfw_concept = (
                self.pipe.safety_checker is not None
                and outputs.nsfw_content_detected[0]
            )
        canvas = []
        for orig, out in zip(image, outputs.images):
            canvas.append(out.resize(orig.size, Image.BILINEAR))
        return canvas, label


class DDIMLoraGeneration(GenerativeMixup):

    pipe = None  # global sharing is a hack to avoid OOM

    def __init__(
        self,
        lora_path: str,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        embed_path: str = None,
        prompt: str = "a photo of a {name}",
        format_name: Callable = format_name,
        guidance_scale: float = 7.5,
        disable_safety_checker: bool = True,
        revision: str = None,
        device="cuda",
        ddim_eta: float = 0.0,
        num_inference_steps: int = 25,
        **kwargs,
    ):

        super(DDIMLoraGeneration, self).__init__()

        if DDIMLoraGeneration.pipe is None:

            PipelineClass = StableDiffusionPipeline

            DDIMLoraGeneration.pipe = PipelineClass.from_pretrained(
                    model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                    torch_dtype=torch.float16,
                ).to(device)

            scheduler = DDIMScheduler.from_config(
                DDIMLoraGeneration.pipe.scheduler.config, local_files_only=True
            )
            self.placeholder2name = None
            self.name2placeholder = None
            if embed_path is not None:
                self.name2placeholder, self.placeholder2name = load_diffmix_embeddings(
                    embed_path,
                    DDIMLoraGeneration.pipe.text_encoder,
                    DDIMLoraGeneration.pipe.tokenizer,
                )
            if lora_path is not None:
                DDIMLoraGeneration.pipe.load_lora_weights(lora_path)
            DDIMLoraGeneration.pipe.scheduler = scheduler

            print(f"successfuly load lora weights from {lora_path}! ! ! ")

            logging.disable_progress_bar()
            self.pipe.set_progress_bar_config(disable=True)

            if disable_safety_checker:
                self.pipe.safety_checker = None

        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.format_name = format_name
        self.ddim_eta = ddim_eta
        self.num_inference_steps = num_inference_steps
        # 预计算特征支持
        self.pre_features_root = os.getenv("FEATURES_PER_CLASS_DIR", None)
        self._pre_features_cache = {}
        # 初始化CLIP为持久对象，避免每次forward重新加载
        self.clip_model = None
        self.clip_processor = None
        self._init_clip(device)

    def _init_clip(self, device):
        try:
            clip_model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model.eval()
        except Exception as e:
            print(f"Warning: CLIP init failed: {e}")

    def _sanitize_class_name(self, name: str) -> str:
        return str(name).replace("/", " ").replace(" ", "_")

    def _load_precomputed_for_class(self, class_name: str, device) -> dict:
        """惰性加载指定类别的预计算特征: {clip, text_clip, text_encoder_hidden_states, pen, fin}。若找不到返回{}"""
        if not self.pre_features_root:
            return {}
        key = self._sanitize_class_name(class_name)
        if key in self._pre_features_cache:
            return self._pre_features_cache[key]
        # 在根目录下搜索以 name_ 开头的类别目录
        pattern = os.path.join(self.pre_features_root, f"{key}_*")
        matches = sorted(glob.glob(pattern))
        if not matches:
            self._pre_features_cache[key] = {}
            return {}
        cls_dir = matches[0]
        # 找到 pt 文件
        pt_candidates = glob.glob(os.path.join(cls_dir, "*_avg_features.pt"))
        if not pt_candidates:
            self._pre_features_cache[key] = {}
            return {}
        try:
            data = torch.load(pt_candidates[0], map_location="cpu")
            clip_t = data.get("clip_features", None)
            text_clip_t = data.get("text_clip_features", None)  # 文本CLIP特征
            text_encoder_t = data.get("text_encoder_hidden_states", None)  # 完整文本编码器输出
            pen_t = data.get("up_block_penultimate_features", None)
            fin_t = data.get("up_block_final_features", None)
            out = {}
            if clip_t is not None:
                out["clip"] = clip_t.to(device)
            if text_clip_t is not None:
                out["text_clip"] = text_clip_t.to(device)  # 文本CLIP特征
            if text_encoder_t is not None:
                out["text_encoder_hidden_states"] = text_encoder_t.to(device)  # 完整文本编码器输出
            if pen_t is not None:
                out["pen"] = pen_t.to(device)
            if fin_t is not None:
                out["fin"] = fin_t.to(device)
            self._pre_features_cache[key] = out
            return out
        except Exception:
            self._pre_features_cache[key] = {}
            return {}

    def _get_target_features(
        self,
        name: str,
        image: Image.Image,
        metadata: dict,
        device,
        vae,
        unet,
        scheduler: DDIMScheduler,
        cond_emb: torch.Tensor,
    ):
        """返回(clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt)"""
        # 优先用预计算特征
        pre_t = self._load_precomputed_for_class(metadata.get("name", name), device)
        
        # 从预计算特征中获取文本CLIP特征
        clip_target_txt_feat = None
        if pre_t and "text_clip" in pre_t:
            clip_target_txt_feat = pre_t["text_clip"]

        # 从预计算特征中获取其他特征
        clip_target_img_feat = None
        up_feats_target = {}
        if pre_t:
            clip_target_img_feat = pre_t.get("clip", None)
            up_feats_target["pen"] = pre_t.get("pen", None)
            up_feats_target["fin"] = pre_t.get("fin", None)

        return clip_target_img_feat, clip_target_txt_feat, up_feats_target.get("pen"), up_feats_target.get("fin")

    def _perform_guidance_step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        i: int,
        cond_emb: torch.Tensor,
        uncond_emb: torch.Tensor,
        vae,
        unet,
        scheduler: DDIMScheduler,
        device,
        clip_target_img_feat,
        clip_target_txt_feat,
        pen_tgt,
        fin_tgt,
        use_clip_losses: bool,
    ):
        # 1) 两分支合并前向（CFG）
        latents_in = torch.cat([latents, latents], dim=0)
        emb_in = torch.cat([uncond_emb, cond_emb], dim=0)
        with torch.no_grad():
            out = unet(latents_in, t, emb_in).sample
            e_uncond, e_text = out.chunk(2)
            e_t = e_uncond + self.guidance_scale * (e_text - e_uncond)

        # 2) pred_x0
        alpha_bar_t = scheduler.alphas_cumprod[i].to(device)
        sqrt_one_minus_at = (1.0 - alpha_bar_t).sqrt()
        pred_x0 = ((latents - sqrt_one_minus_at * e_t) / alpha_bar_t.sqrt()).detach()
        pred_x0.requires_grad_(True)

        # 3) 解码 -> CLIP图像特征
        x0_vae = (pred_x0 / vae.config.scaling_factor).to(unet.dtype)
        with torch.no_grad():
            recons = vae.decode(x0_vae).sample
        recons = (recons / 2 + 0.5).clamp(0, 1)
        recons_np = (recons * 255).to(torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
        pil_images = [Image.fromarray(img) for img in recons_np]
        if self.clip_model is not None and self.clip_processor is not None:
            clip_inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            clip_img_feat = self.clip_model.get_image_features(**clip_inputs)
            clip_img_feat = clip_img_feat / (clip_img_feat.norm(dim=-1, keepdim=True) + 1e-6)
        else:
            clip_img_feat = None

        # 4) 当前UNet两层（可微，输出[B,C]）
        features_cur = {"pen": None, "fin": None}
        def hook_pen(m, ii, o):
            features_cur["pen"] = o
        def hook_fin(m, ii, o):
            features_cur["fin"] = o
        num_up = len(unet.up_blocks)
        h1 = unet.up_blocks[num_up - 2].register_forward_hook(hook_pen)
        h2 = unet.up_blocks[num_up - 1].register_forward_hook(hook_fin)
        xt_from_x0 = pred_x0.to(unet.dtype)   #这里我动了，之前是alpha_bar_t.sqrt() * pred_x0
        _ = unet(xt_from_x0, t, cond_emb)
        h1.remove(); h2.remove()
        def gap_var(x):
            return x.mean(dim=[2, 3])
        up_pen_cur = gap_var(features_cur["pen"]) if features_cur["pen"] is not None else None
        up_fin_cur = gap_var(features_cur["fin"]) if features_cur["fin"] is not None else None  #这里做了平均

        # 5) 四项内积式损失
        losses = []
        if use_clip_losses:
            if clip_img_feat is not None and clip_target_img_feat is not None:
                delta = (clip_img_feat.detach() - clip_target_img_feat.detach().to(clip_img_feat.dtype))
                losses.append(-0.5*(clip_img_feat * delta).sum())
            if clip_img_feat is not None and clip_target_txt_feat is not None:
                delta = (clip_img_feat.detach() - clip_target_txt_feat.detach().to(clip_img_feat.dtype))
                losses.append(-0.5*(clip_img_feat * delta).sum())
        # disable guidance loss from the penultimate upsampling layer
        # if up_pen_cur is not None and pen_tgt is not None:
        #     pen_tgt = pen_tgt.detach().to(up_pen_cur.device, dtype=up_pen_cur.dtype)
        #     delta = (up_pen_cur - pen_tgt)
        #     losses.append(-(up_pen_cur * delta).sum())
        if up_fin_cur is not None and fin_tgt is not None:
            fin_tgt = fin_tgt.detach().to(up_fin_cur.device, dtype=up_fin_cur.dtype)
            delta = (up_fin_cur.detach() - fin_tgt)
            losses.append(-0.5*(up_fin_cur * delta).sum())

        if len(losses) > 0:
            total_loss = sum(losses)
            total_loss.backward()
            grad = pred_x0.grad
            if grad is not None:
                e_t = e_t - sqrt_one_minus_at * grad.detach()
            pred_x0.grad = None

        return e_t

    def forward(
        self,
        image: Image.Image,
        label: int,
        metadata: dict,
        strength: float = 0.5,
        resolution=512,
    ) -> Tuple[Image.Image, int]:

        device = self.pipe.device
        weight_dtype = self.pipe.unet.dtype
        dtype = weight_dtype

        name = metadata.get("name", "")
        if self.name2placeholder is not None:
            name = self.name2placeholder[name]
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        batch_size = max(1, len(image) if isinstance(image, (list, tuple)) else 1)
        height = resolution
        width = resolution

        # 1) 文本条件与无条件嵌入（CFG）
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        text_inputs = tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_inputs = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            cond_emb = text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)
            uncond_emb = text_encoder(uncond_inputs.input_ids.to(device))[0].to(dtype=weight_dtype)

        # 2) 目标特征（预计算/动态）
        vae = self.pipe.vae
        unet = self.pipe.unet
        scheduler: DDIMScheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=device)
        t0 = scheduler.timesteps[0]
        clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt = self._get_target_features(
            name, image, metadata, device, vae, unet, scheduler, cond_emb
        )

        # 4) 采样初始化
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            device=device, dtype=weight_dtype
        )
        # 5) DDIM迭代
        total_steps = len(scheduler.timesteps)
        for i, t in enumerate(scheduler.timesteps):
            use_clip_losses = i >= (total_steps - 5)
            e_t = self._perform_guidance_step(
                latents, t, i, cond_emb, uncond_emb, vae, unet, scheduler, device,
                clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt,
                use_clip_losses
            )

            # 5.6 DDIM更新
            latents = scheduler.step(e_t, t, latents, eta=self.ddim_eta).prev_sample

        # 6) 最终解码
        x_final = latents / vae.config.scaling_factor
        with torch.no_grad():
            imgs = vae.decode(x_final.to(dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu()
        pil_list = []
        for i in range(imgs.shape[0]):
            arr = (imgs[i].permute(1, 2, 0).numpy() * 255).astype('uint8')
            pil_list.append(Image.fromarray(arr))
        return pil_list, label