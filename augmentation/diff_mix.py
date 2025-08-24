import os
from typing import Callable, Tuple

import torch
from PIL import Image
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
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

        #print(prompt)

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

            PipelineClass = StableDiffusionImg2ImgPipeline
            # 1. 单独加载 image_encoder 和 feature_extractor
                # 通常使用 OpenAI 的标准 CLIP vision 模型
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to(device)

            # feature_extractor 是 image_encoder 的预处理器
            feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            DDIMLoraGeneration.pipe = PipelineClass.from_pretrained(
                    model_path,
                use_auth_token=True,
                revision=revision,
                local_files_only=True,
                    torch_dtype=torch.float16,
                    image_encoder=image_encoder, # <--- 在这里传入
            feature_extractor=feature_extractor # <--- 在这里传入
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

        # 复用pipeline中的CLIP组件，避免额外显存
        self.image_encoder = getattr(DDIMLoraGeneration.pipe, "image_encoder", None)
        self.feature_extractor = getattr(DDIMLoraGeneration.pipe, "feature_extractor", None)


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
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
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
        emb_in = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        latents_in.requires_grad_(True)
        
        with torch.no_grad():
            out = unet(latents_in, t, emb_in).sample
            e_uncond, e_text = out.chunk(2)
            e_t = e_uncond + self.guidance_scale * (e_text - e_uncond)
        # 5) 四项内积式损失
        losses = []
        #print(self.image_encoder,self.feature_extractor)
        #import pdb; pdb.set_trace()
        # 计算clip_img_feat1（基于latents）
        if self.image_encoder is not None and self.feature_extractor is not None and use_clip_losses:
            # 从latents解码得到图像 - 使用e_text而不是e_t来确保维度匹配
            alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
            sqrt_one_minus_at = (1.0 - alpha_bar_t).sqrt()
            # 使用e_text（positive prompt分支）而不是e_t
            pred_x0_from_latents = latents.detach() 
            
            x0_vae_from_latents = (pred_x0_from_latents / vae.config.scaling_factor).to(unet.dtype)
            with torch.no_grad():
                recons_from_latents = vae.decode(x0_vae_from_latents).sample
            pred_x0_from_latents.requires_grad_(True)
            recons_from_latents = (recons_from_latents / 2 + 0.5).clamp(0, 1)
            recons_np_from_latents = (recons_from_latents * 255).to(torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
            pil_images_from_latents = [Image.fromarray(img) for img in recons_np_from_latents]
            
            fea_from_latents = self.feature_extractor(images=pil_images_from_latents, return_tensors="pt")
            pixel_values_from_latents = fea_from_latents.pixel_values.to(device)
            # 将输入的 pixel_values 转换为与 image_encoder 相同的类型 (float16)
            pixel_values_from_latents = pixel_values_from_latents.to(self.image_encoder.dtype)
            with torch.no_grad():
                img_embeds_from_latents = self.image_encoder(pixel_values_from_latents).image_embeds
            clip_img_feat1 = img_embeds_from_latents / (img_embeds_from_latents.norm(dim=-1, keepdim=True) + 1e-6)
            
            # 注册hook来获取特征
            features_cur1 = {"pen": None, "fin": None}
            def hook_pen1(m, ii, o):
                features_cur1["pen"] = o
            def hook_fin1(m, ii, o):
                features_cur1["fin"] = o
            
            num_up = len(unet.up_blocks)
            h1_1 = unet.up_blocks[num_up - 2].register_forward_hook(hook_pen1)
            h2_1 = unet.up_blocks[num_up - 1].register_forward_hook(hook_fin1)
            
            
            out_features = unet(pred_x0_from_latents, t, prompt_embeds).sample
            
            # 移除hook
            h1_1.remove()
            h2_1.remove()
            
            # 计算特征 - 只取有条件分支（positive prompt）的特征
            def gap_var(x):
                return x.mean(dim=[2, 3])
            
            # 从拼接的特征中只取后半部分（positive prompt分支）
            if features_cur1["pen"] is not None:
                pen_positive = features_cur1["pen"] # 取后半部分
                up_pen_cur1 = gap_var(pen_positive)
            else:
                up_pen_cur1 = None
                
            if features_cur1["fin"] is not None:
                fin_positive = features_cur1["fin"]  # 取后半部分
                up_fin_cur1 = gap_var(fin_positive)
            else:
                up_fin_cur1 = None
        else:
            clip_img_feat1 = None

        if use_clip_losses:
            with torch.no_grad():
        # 2) pred_x0
                alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
                sqrt_one_minus_at = (1.0 - alpha_bar_t).sqrt()
                pred_x0 = ((latents - sqrt_one_minus_at * e_t) / alpha_bar_t.sqrt()).clone().detach()
            
                # 3) 解码 -> CLIP图像特征
                x0_vae = (pred_x0 / vae.config.scaling_factor).to(unet.dtype)
                #print(pred_x0_from_latents.requires_grad)
                with torch.no_grad():
                    recons = vae.decode(x0_vae).sample
                #print(pred_x0_from_latents.requires_grad)
                recons = (recons / 2 + 0.5).clamp(0, 1)
                recons_np = (recons * 255).to(torch.uint8).detach().cpu().permute(0, 2, 3, 1).numpy()
                pil_images = [Image.fromarray(img) for img in recons_np]
                if self.image_encoder is not None and self.feature_extractor is not None:
                    fea = self.feature_extractor(images=pil_images, return_tensors="pt")
                    pixel_values = fea.pixel_values.to(device)
                    pixel_values = pixel_values.to(self.image_encoder.dtype)
                    with torch.no_grad():
                        img_embeds = self.image_encoder(pixel_values).image_embeds
                    clip_img_feat = img_embeds / (img_embeds.norm(dim=-1, keepdim=True) + 1e-6)
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
                _ = unet(xt_from_x0, t, prompt_embeds)
                h1.remove(); h2.remove()
                def gap_var(x):
                    return x.mean(dim=[2, 3])
                up_pen_cur = gap_var(features_cur["pen"]) if features_cur["pen"] is not None else None
                up_fin_cur = gap_var(features_cur["fin"]) if features_cur["fin"] is not None else None  #这里做了平均
            
                
            
            # 在最后5步中启用CLIP损失进行梯度引导
            if clip_img_feat is not None and clip_target_img_feat is not None:
                delta = (clip_img_feat.detach() - clip_target_img_feat.detach().to(clip_img_feat.dtype))
                losses.append(0.5*(clip_img_feat1 * delta).sum())
            if clip_img_feat is not None and clip_target_txt_feat is not None:
                delta = (clip_img_feat.detach() - clip_target_txt_feat.detach().to(clip_img_feat.dtype))
                losses.append(0.5*(clip_img_feat1 * delta).sum())
            # 在最后5步中也启用UNet特征损失进行梯度引导
            if up_fin_cur is not None and fin_tgt is not None:
                fin_tgt = fin_tgt.detach().to(up_fin_cur.device, dtype=up_fin_cur.dtype)
                delta = (up_fin_cur.detach() - fin_tgt)
                losses.append(0.5*(up_fin_cur1 * delta).sum())
        # 注释掉：不在最后5步时不使用任何梯度引导
        # if up_fin_cur is not None and fin_tgt is not None:
        #     fin_tgt = fin_tgt.detach().to(up_fin_cur.device, dtype=up_fin_cur.dtype)
        #     delta = (up_fin_cur.detach() - fin_tgt)
        #     losses.append(0.5*(up_fin_cur * delta).sum())
        #print(pred_x0_from_latents.requires_grad)
        if len(losses) > 0:
            total_loss = sum(losses)
            # 仅对 pred_x0 求梯度，避免对同一计算图多次 backward
            #print(pred_x0_from_latents.requires_grad)
            grad = torch.autograd.grad(
                total_loss,
                pred_x0_from_latents,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            #print(grad)
            if grad is not None:
                e_t = e_t - sqrt_one_minus_at * grad.detach()
            pred_x0_from_latents.grad = None

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

        # 准备输入图像
        canvas = [img.resize((resolution, resolution), Image.BILINEAR) for img in image]
        
        # 构建提示词
        name = metadata.get("name", "")
        if self.name2placeholder is not None:
            name = self.name2placeholder[name]
        if metadata.get("super_class", None) is not None:
            name = name + " " + metadata.get("super_class", "")
        prompt = self.prompt.format(name=name)

        #print(prompt)
        #print(prompt, flush=True)

        # 步骤一：使用 Pipeline 的标准方法进行输入准备
        # 1. 文本编码 - 使用 Pipeline 的 encode_prompt 方法
        do_classifier_free_guidance = self.guidance_scale > 1.0
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=len(canvas),
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
        )
        # 若未启用CFG，negative_prompt_embeds可能为None；为后续拼接提供回退
        if negative_prompt_embeds is None:
            negative_prompt_embeds = prompt_embeds
        
        # 2. 图像预处理 - 使用 Pipeline 的 image_processor
        processed_image = self.pipe.image_processor.preprocess(canvas)
        
        # 3. 设置和获取时间步
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(
            self.num_inference_steps, strength, device
        )
        
        # 4. 准备潜变量 - 使用 Pipeline 的 prepare_latents 方法
        latents = self.pipe.prepare_latents(
            image=processed_image,
            timestep=timesteps[:1].repeat(len(canvas)),
            batch_size=len(canvas),
            num_images_per_prompt=1,
            dtype=weight_dtype,
            device=device,
            generator=None,
        )

        # 记录原始尺寸，避免变量覆盖导致获取到Tensor.size方法
        original_sizes = [img.size for img in image]

        # 步骤二：目标特征准备（预计算/动态）
        clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt = None, None, None, None
        if self.pre_features_root:
            clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt = self._get_target_features(
                name, image, metadata, device, self.pipe.vae, self.pipe.unet, self.pipe.scheduler, prompt_embeds)
            
        # 步骤三：自定义采样循环
        for j, t in enumerate(timesteps):
            # 只在最后5步启用CLIP损失
            use_clip_losses = j >= len(timesteps) - 5
            
            # 调用自定义引导步骤
            e_t = self._perform_guidance_step(
                latents, t, j, prompt_embeds, negative_prompt_embeds, 
                self.pipe.vae, self.pipe.unet, self.pipe.scheduler, device,
                clip_target_img_feat, clip_target_txt_feat, pen_tgt, fin_tgt,
                use_clip_losses
            )

            # DDIM 更新
            latents = self.pipe.scheduler.step(e_t, t, latents, eta=self.ddim_eta).prev_sample

        # 步骤四：使用 Pipeline 的标准后处理方法
        # 解码潜变量
        with torch.no_grad():
            decoded = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
            
            # 安全检查（如果启用）
            if self.pipe.safety_checker is not None:
                decoded, has_nsfw_concept = self.pipe.run_safety_checker(decoded, device, weight_dtype)
            else:
                has_nsfw_concept = None

            # 后处理 - 使用 Pipeline 的 image_processor.postprocess
            if has_nsfw_concept is None:
                do_denormalize = [True] * decoded.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            processed_images = self.pipe.image_processor.postprocess(
                decoded, output_type="pil", do_denormalize=do_denormalize
            )

        # 与 mix 一致：按输入原图尺寸 resize 返回
        resized = []
        for sz, out in zip(original_sizes, processed_images):
            resized.append(out.resize(sz, Image.BILINEAR))
        
        return resized, label