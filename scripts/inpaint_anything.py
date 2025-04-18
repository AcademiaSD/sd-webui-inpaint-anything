# Standard library imports for general utilities
import gc  # Garbage collection for memory management
import math  # Mathematical operations
import os  # Operating system interactions
import platform  # Platform-specific settings
import random  # Random number generation
import re  # Regular expressions for string manipulation
import traceback  # Exception traceback printing

# Third-party library imports for image processing and machine learning
import cv2  # OpenCV for image processing
import gradio as gr  # Gradio for building the UI
from gradio import ImageEditor, Brush  # Specific Gradio components
import numpy as np  # Numerical operations on arrays
import torch  # PyTorch for deep learning
from diffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    StableDiffusionInpaintPipeline,
)  # Diffusers library for Stable Diffusion pipelines
from modules import devices, script_callbacks, shared  # Web UI shared modules
from modules.processing import create_infotext, process_images  # Image processing utilities
from modules.sd_models import get_closet_checkpoint_match  # Model checkpoint utilities
from modules.sd_samplers import samplers_for_img2img  # Samplers for img2img
from PIL import Image, ImageFilter, ImageOps  # PIL for image manipulation
from PIL.PngImagePlugin import PngInfo  # PNG metadata handling
from torch.hub import download_url_to_file  # Model downloading utility
from torchvision import transforms  # Image transformations

# Custom imports specific to this extension
import inpalib  # Inpainting library functions
from ia_check_versions import ia_check_versions  # Version checking utilities
from ia_config import (
    IAConfig,
    get_ia_config_index,
    get_webui_setting,
    set_ia_config,
    setup_ia_config_ini,
)  # Configuration management
from ia_file_manager import (
    IAFileManager,
    download_model_from_hf,
    ia_file_manager,
)  # File management utilities
from ia_logging import draw_text_image, ia_logging  # Logging utilities
from ia_threading import (
    async_post_reload_model_weights,
    await_backup_reload_ckpt_info,
    await_pre_reload_model_weights,
    clear_cache_decorator,
    offload_reload_decorator,
)  # Threading and caching decorators
from ia_ui_items import (
    get_cleaner_model_ids,
    get_inp_model_ids,
    get_inp_webui_model_ids,
    get_padding_mode_names,
    get_sam_model_ids,
    get_sampler_names,
)  # UI item retrieval functions
from ia_webui_controlnet import (
    backup_alwayson_scripts,
    clear_controlnet_cache,
    disable_all_alwayson_scripts,
    disable_alwayson_scripts_wo_cn,
    find_controlnet,
    get_controlnet_args_to,
    get_max_args_to,
    get_sd_img2img_processing,
    restore_alwayson_scripts,
)  # ControlNet integration
from lama_cleaner.model_manager import ModelManager  # Cleaner model management
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler  # Cleaner configuration

# Platform-specific configurations
if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for macOS
if platform.system() == "Windows":
    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"  # Disable Triton on Windows for compatibility

# Global state dictionary for managing segmentation and inpainting data
sam_state = {
    "sam_masks": None,  # Stores segmentation masks from Segment Anything
    "mask_image": None,  # Current mask image being processed
    "controlnet": None,  # ControlNet instance if available
    "original_image": None,  # Original input image
    "padding_mask": None,  # Mask for padded areas
}


@clear_cache_decorator
def download_segment_anything_model(model_id: str) -> str:
    """
    Downloads the specified Segment Anything (SAM) model from its source URL.

    Args:
        model_id (str): Identifier of the SAM model to download.

    Returns:
        str: Status message indicating success or failure.
    """
    # Define model source URLs based on model type
    if "_hq_" in model_id:
        url = f"https://huggingface.co/Uminosachi/sam-hq/resolve/main/{model_id}"
    elif "FastSAM" in model_id:
        url = f"https://huggingface.co/Uminosachi/FastSAM/resolve/main/{model_id}"
    elif "mobile_sam" in model_id:
        url = f"https://huggingface.co/Uminosachi/MobileSAM/resolve/main/{model_id}"
    elif "sam2_" in model_id:
        url = f"https://dl.fbaipublicfiles.com/segment_anything_2/072824/{model_id}"
    else:
        url = f"https://dl.fbaipublicfiles.com/segment_anything/{model_id}"

    # Define local path for model storage
    model_path = os.path.join(ia_file_manager.models_dir, model_id)

    # Check if model already exists locally
    if not os.path.isfile(model_path):
        try:
            download_url_to_file(url, model_path)
            return IAFileManager.DOWNLOAD_COMPLETE
        except Exception as error:
            ia_logging.error(f"Download failed: {str(error)}")
            return str(error)
    return "Model already exists"


def save_mask_image(mask_image: np.ndarray, save_mask_enabled: bool = False) -> None:
    """
    Saves the mask image to the outputs directory if enabled.

    Args:
        mask_image (np.ndarray): The mask image to save.
        save_mask_enabled (bool, optional): If True, saves the mask image. Defaults to False.

    Returns:
        None
    """
    if save_mask_enabled:
        filename = f"{ia_file_manager.savename_prefix}_created_mask.png"
        save_path = os.path.join(ia_file_manager.outputs_dir, filename)
        Image.fromarray(mask_image).save(save_path)
        ia_logging.info(f"Mask saved to {save_path}")


@clear_cache_decorator
def handle_input_image_upload(
    input_image: np.ndarray,
    segment_anything_image: dict,
    selected_mask: dict
) -> tuple:
    """
    Processes the uploaded input image and updates related components.

    Args:
        input_image (np.ndarray): The uploaded input image.
        segment_anything_image (dict): Current state of the Segment Anything image editor.
        selected_mask (dict): Current state of the selected mask image editor.

    Returns:
        tuple: Updates for segment_anything_image, selected_mask, and run_segment_anything_button.
    """
    global sam_state
    sam_state["original_image"] = input_image
    sam_state["padding_mask"] = None

    # Initialize mask_image if it doesn't exist or doesn't match input_image shape
    if (
        sam_state["mask_image"] is None
        or not isinstance(sam_state["mask_image"], np.ndarray)
        or sam_state["mask_image"].shape != input_image.shape
    ):
        sam_state["mask_image"] = np.zeros_like(input_image, dtype=np.uint8)

    # Blend input image with current mask for display
    blended_image = cv2.addWeighted(input_image, 0.5, sam_state["mask_image"], 0.5, 0)

    # Update Segment Anything image
    if (
        segment_anything_image is None
        or not isinstance(segment_anything_image, dict)
        or "image" not in segment_anything_image
    ):
        sam_state["sam_masks"] = None
        segment_anything_update = np.zeros_like(input_image, dtype=np.uint8)
    elif segment_anything_image["composite"][:, :, :3].shape == input_image.shape:
        segment_anything_update = gr.update()
    else:
        sam_state["sam_masks"] = None
        segment_anything_update = gr.update(value=np.zeros_like(input_image, dtype=np.uint8))

    # Update selected mask image
    if (
        selected_mask is None
        or not isinstance(selected_mask, dict)
        or "image" not in selected_mask
    ):
        selected_mask_update = blended_image
    elif (
        selected_mask["composite"][:, :, :3].shape == blended_image.shape
        and np.all(selected_mask["composite"][:, :, :3] == blended_image)
    ):
        selected_mask_update = gr.update()
    else:
        selected_mask_update = gr.update(value=blended_image)

    return segment_anything_update, selected_mask_update, gr.update(interactive=True)


@clear_cache_decorator
def apply_padding(
    input_image: np.ndarray,
    scale_width: float,
    scale_height: float,
    left_right_balance: float,
    top_bottom_balance: float,
    padding_mode: str = "edge"
) -> tuple:
    """
    Applies padding to the input image based on specified parameters.

    Args:
        input_image (np.ndarray): The input image to pad.
        scale_width (float): Width scaling factor (1.0 to 1.5).
        scale_height (float): Height scaling factor (1.0 to 1.5).
        left_right_balance (float): Balance between left and right padding (0.0 to 1.0).
        top_bottom_balance (float): Balance between top and bottom padding (0.0 to 1.0).
        padding_mode (str, optional): Padding mode ('edge', 'constant'). Defaults to "edge".

    Returns:
        tuple: Padded image and status message.
    """
    global sam_state
    if input_image is None or sam_state["original_image"] is None:
        sam_state["original_image"] = None
        sam_state["padding_mask"] = None
        return None, "Input image not found"

    original_image = sam_state["original_image"]
    height, width = original_image.shape[:2]
    padded_width = int(width * scale_width)
    padded_height = int(height * scale_height)
    ia_logging.info(f"Padding image from ({height}, {width}) to ({padded_height}, {padded_width})")

    # Calculate padding sizes
    pad_width_size = padded_width - width
    pad_height_size = padded_height - height
    pad_left = int(pad_width_size * left_right_balance)
    pad_right = pad_width_size - pad_left
    pad_top = int(pad_height_size * top_bottom_balance)
    pad_bottom = pad_height_size - pad_top

    # Apply padding to the image
    padding = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    if padding_mode == "constant":
        fill_value = get_webui_setting("inpaint_anything_padding_fill", 127)
        padded_image = np.pad(original_image, padding, mode=padding_mode, constant_values=fill_value)
    else:
        padded_image = np.pad(original_image, padding, mode=padding_mode)

    # Create padding mask
    mask_padding = [(pad_top, pad_bottom), (pad_left, pad_right)]
    padding_mask = np.zeros((height, width), dtype=np.uint8)
    padding_mask = np.pad(padding_mask, mask_padding, mode="constant", constant_values=255)
    sam_state["padding_mask"] = {"segmentation": padding_mask.astype(bool)}

    return padded_image, "Padding completed"


@offload_reload_decorator
@clear_cache_decorator
def run_segment_anything(
    input_image: np.ndarray,
    model_id: str,
    segment_anything_image: dict,
    anime_style_enabled: bool = False
) -> tuple:
    """
    Runs the Segment Anything model on the input image.

    Args:
        input_image (np.ndarray): The input image to segment.
        model_id (str): Identifier of the SAM model to use.
        segment_anything_image (dict): Current state of the Segment Anything image editor.
        anime_style_enabled (bool, optional): If True, optimizes for anime-style images. Defaults to False.

    Returns:
        tuple: Updated segment_anything_image and status message.
    """
    global sam_state
    if not inpalib.sam_file_exists(model_id):
        return (
            None if segment_anything_image is None else gr.update(),
            f"{model_id} not found, please download"
        )
    if input_image is None:
        return (
            None if segment_anything_image is None else gr.update(),
            "Input image not found"
        )

    set_ia_config(IAConfig.KEYS.SAM_MODEL_ID, model_id, IAConfig.SECTIONS.USER)
    if sam_state["sam_masks"] is not None:
        sam_state["sam_masks"] = None
        gc.collect()

    ia_logging.info(f"Processing input_image: {input_image.shape} {input_image.dtype}")

    try:
        # Generate and sort masks
        sam_masks = inpalib.generate_sam_masks(input_image, model_id, anime_style_enabled)
        sam_masks = inpalib.sort_masks_by_area(sam_masks)
        sam_masks = inpalib.insert_mask_to_sam_masks(sam_masks, sam_state["padding_mask"])
        segmented_image = inpalib.create_seg_color_image(input_image, sam_masks)
        sam_state["sam_masks"] = sam_masks
    except Exception as error:
        ia_logging.error(f"Segment Anything failed: {str(error)}")
        print(traceback.format_exc())
        return (
            None if segment_anything_image is None else gr.update(),
            "Segment Anything failed"
        )

    # Update Gradio component based on current state
    if (
        segment_anything_image is None
        or not isinstance(segment_anything_image, dict)
        or "image" not in segment_anything_image
    ):
        return segmented_image, "Segment Anything completed"
    if segment_anything_image["composite"][:, :, :3].shape == segmented_image.shape and np.all(
        segment_anything_image["composite"][:, :, :3] == segmented_image
    ):
        return gr.update(), "Segment Anything completed"
    return gr.update(value=segmented_image), "Segment Anything completed"


@clear_cache_decorator
def select_mask(
    input_image: np.ndarray,
    segment_anything_image: dict,
    invert_mask_enabled: bool,
    ignore_black_enabled: bool,
    selected_mask: dict
) -> gr.update:
    """
    Creates a mask based on user selection in the Segment Anything image editor.

    Args:
        input_image (np.ndarray): The original input image.
        segment_anything_image (dict): State of the Segment Anything image editor.
        invert_mask_enabled (bool): If True, inverts the generated mask.
        ignore_black_enabled (bool): If True, ignores black areas in mask creation.
        selected_mask (dict): Current state of the selected mask image editor.

    Returns:
        gr.update: Updated selected_mask Gradio component.
    """
    global sam_state
    if sam_state["sam_masks"] is None or segment_anything_image is None:
        return None if selected_mask is None else gr.update()

    sam_masks = sam_state["sam_masks"]

    # Extract mask from Segment Anything image editor layers
    if "layers" in segment_anything_image and segment_anything_image["layers"]:
        layers = segment_anything_image["layers"]
        alpha_channels = [layer[:, :, 3] for layer in layers]
        max_alpha = np.max(alpha_channels, axis=0)
        mask = (max_alpha > 0).astype(np.uint8) * 255
        mask = mask[:, :, np.newaxis]
    else:
        height, width = segment_anything_image["background"].shape[:2]
        mask = np.zeros((height, width, 1), dtype=np.uint8)

    try:
        # Create and optionally invert the mask
        mask_image = inpalib.create_mask_image(mask, sam_masks, ignore_black_enabled)
        if invert_mask_enabled:
            mask_image = inpalib.invert_mask(mask_image)
        sam_state["mask_image"] = mask_image
    except Exception as error:
        ia_logging.error(f"Mask selection failed: {str(error)}")
        print(traceback.format_exc())
        return None if selected_mask is None else gr.update()

    # Blend mask with input image if dimensions match
    blended_image = (
        cv2.addWeighted(input_image, 0.5, mask_image, 0.5, 0)
        if input_image is not None and input_image.shape == mask_image.shape
        else mask_image
    )

    # Update Gradio component
    if (
        selected_mask is None
        or not isinstance(selected_mask, dict)
        or "image" not in selected_mask
    ):
        return blended_image
    if (
        selected_mask["composite"][:, :, :3].shape == blended_image.shape
        and np.all(selected_mask["composite"][:, :, :3] == blended_image)
    ):
        return gr.update()
    return gr.update(value=blended_image)


@clear_cache_decorator
def expand_mask(
    input_image: np.ndarray,
    selected_mask: dict,
    expand_iterations: int = 1
) -> gr.update:
    """
    Expands the mask region by dilating it.

    Args:
        input_image (np.ndarray): The original input image.
        selected_mask (dict): Current state of the selected mask image editor.
        expand_iterations (int, optional): Number of dilation iterations (1 to 100). Defaults to 1.

    Returns:
        gr.update: Updated selected_mask Gradio component.
    """
    global sam_state
    if sam_state["mask_image"] is None or selected_mask is None:
        return None

    mask_image = sam_state["mask_image"]
    expand_iterations = int(np.clip(expand_iterations, 1, 100))
    expanded_mask = cv2.dilate(mask_image, np.ones((3, 3), dtype=np.uint8), iterations=expand_iterations)
    sam_state["mask_image"] = expanded_mask

    # Blend with input image if dimensions match
    blended_image = (
        cv2.addWeighted(input_image, 0.5, expanded_mask, 0.5, 0)
        if input_image is not None and input_image.shape == expanded_mask.shape
        else expanded_mask
    )

    # Update Gradio component
    if (
        selected_mask["composite"][:, :, :3].shape == blended_image.shape
        and np.all(selected_mask["composite"][:, :, :3] == blended_image)
    ):
        return gr.update()
    return gr.update(value=blended_image)


@clear_cache_decorator
def trim_mask_by_sketch(
    input_image: np.ndarray,
    selected_mask: dict
) -> gr.update:
    """
    Trims the mask based on the user's sketch in the selected mask image editor.

    Args:
        input_image (np.ndarray): The original input image.
        selected_mask (dict): State of the selected mask image editor.

    Returns:
        gr.update: Updated selected_mask Gradio component or None if invalid.
    """
    global sam_state
    if sam_state["mask_image"] is None or selected_mask is None:
        return None

    current_mask = sam_state["mask_image"]

    # Extract sketch mask from layers
    if "layers" in selected_mask and selected_mask["layers"]:
        layers = selected_mask["layers"]
        alpha_channels = [layer[:, :, 3] for layer in layers]
        max_alpha = np.max(alpha_channels, axis=0)
        sketch_mask = (max_alpha > 0).astype(np.uint8) * 255
        trim_mask = np.logical_not(sketch_mask).astype(np.uint8)  # Invert mask
    else:
        height, width = selected_mask["background"].shape[:2]
        trim_mask = np.ones((height, width), dtype=np.uint8) * 255  # No trim

    # Apply trimming
    new_mask = current_mask * trim_mask[:, :, np.newaxis]
    sam_state["mask_image"] = new_mask

    # Blend with input image
    blended_image = (
        cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)
        if input_image is not None and input_image.shape == new_mask.shape
        else new_mask
    )

    # Update Gradio component
    if selected_mask["composite"].shape[:2] == blended_image.shape[:2]:
        return gr.update(value=blended_image)
    return gr.update()


@clear_cache_decorator
def add_mask_by_sketch(
    input_image: np.ndarray,
    selected_mask: dict
) -> gr.update:
    """
    Adds to the mask based on the user's sketch in the selected mask image editor.

    Args:
        input_image (np.ndarray): The original input image.
        selected_mask (dict): State of the selected mask image editor.

    Returns:
        gr.update: Updated selected_mask Gradio component or None if invalid.
    """
    global sam_state
    if sam_state["mask_image"] is None or selected_mask is None:
        return None

    current_mask = sam_state["mask_image"]

    # Extract sketch mask from layers
    if "layers" in selected_mask and selected_mask["layers"]:
        layers = selected_mask["layers"]
        alpha_channels = [layer[:, :, 3] for layer in layers]
        max_alpha = np.max(alpha_channels, axis=0)
        sketch_mask = (max_alpha > 0).astype(np.uint8) * 255
    else:
        height, width = selected_mask["background"].shape[:2]
        sketch_mask = np.zeros((height, width), dtype=np.uint8)

    # Add sketch mask to current mask
    new_mask = np.clip(current_mask + sketch_mask[:, :, np.newaxis], 0, 255).astype(np.uint8)
    sam_state["mask_image"] = new_mask

    # Blend with input image
    blended_image = (
        cv2.addWeighted(input_image, 0.5, new_mask, 0.5, 0)
        if input_image is not None and input_image.shape == new_mask.shape
        else new_mask
    )

    # Update Gradio component
    if selected_mask["composite"].shape[:2] == blended_image.shape[:2]:
        return gr.update(value=blended_image)
    return gr.update()


def auto_resize_to_pil(
    input_image: np.ndarray,
    mask_image: np.ndarray
) -> tuple:
    """
    Resizes the input image and mask to dimensions compatible with the pipeline.

    Args:
        input_image (np.ndarray): The input image.
        mask_image (np.ndarray): The mask image.

    Returns:
        tuple: Resized PIL images (input_image, mask_image).
    """
    init_image = Image.fromarray(input_image).convert("RGB")
    mask_image = Image.fromarray(mask_image).convert("RGB")
    assert init_image.size == mask_image.size, "Image and mask sizes do not match"
    width, height = init_image.size

    # Ensure dimensions are multiples of 8
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    if new_width < width or new_height < height:
        scale = min(new_width / width, new_height / height)
        resize_height = int(height * scale + 0.5)
        resize_width = int(width * scale + 0.5)
        if height != resize_height or width != resize_width:
            ia_logging.info(f"Resizing from ({height}, {width}) to ({resize_height}, {resize_width})")
            init_image = transforms.functional.resize(
                init_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS
            )
            mask_image = transforms.functional.resize(
                mask_image, (resize_height, resize_width), transforms.InterpolationMode.LANCZOS
            )
        if resize_height != new_height or resize_width != new_width:
            ia_logging.info(f"Cropping from ({resize_height}, {resize_width}) to ({new_height}, {new_width})")
            init_image = transforms.functional.center_crop(init_image, (new_height, new_width))
            mask_image = transforms.functional.center_crop(mask_image, (new_height, new_width))

    return init_image, mask_image


@offload_reload_decorator
@clear_cache_decorator
def run_inpaint(
    input_image: np.ndarray,
    selected_mask: dict,
    prompt: str,
    negative_prompt: str,
    sampling_steps: int,
    guidance_scale: float,
    seed: int,
    model_id: str,
    save_mask_enabled: bool,
    composite_enabled: bool,
    sampler_name: str = "DDIM",
    iteration_count: int = 1
):
    """
    Performs inpainting using the Stable Diffusion inpainting pipeline.

    Args:
        input_image (np.ndarray): The input image.
        selected_mask (dict): State of the selected mask image editor.
        prompt (str): Positive prompt for inpainting.
        negative_prompt (str): Negative prompt for inpainting.
        sampling_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale for the model.
        seed (int): Random seed (-1 for random).
        model_id (str): Identifier of the inpainting model.
        save_mask_enabled (bool): If True, saves the mask image.
        composite_enabled (bool): If True, composites the result with the original image.
        sampler_name (str, optional): Sampler to use. Defaults to "DDIM".
        iteration_count (int, optional): Number of iterations. Defaults to 1.

    Yields:
        tuple: List of output images and remaining iterations.
    """
    global sam_state
    if input_image is None or sam_state["mask_image"] is None or selected_mask is None:
        ia_logging.error("Input image or mask missing")
        return

    mask_image = sam_state["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("Input image and mask dimensions do not match")
        return

    set_ia_config(IAConfig.KEYS.INP_MODEL_ID, model_id, IAConfig.SECTIONS.USER)
    save_mask_image(mask_image, save_mask_enabled)

    # Load model with offline support
    offline_mode = get_webui_setting("inpaint_anything_offline_inpainting", False)
    local_files_only = download_model_from_hf(model_id, local_files_only=True) == IAFileManager.DOWNLOAD_COMPLETE
    torch_dtype = torch.float32 if platform.system() == "Darwin" or devices.device == devices.cpu else torch.float16

    try:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, local_files_only=local_files_only, use_safetensors=True
        )
    except Exception as error:
        if not offline_mode:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, use_safetensors=True
            )
        else:
            ia_logging.error(f"Model loading failed: {str(error)}")
            return

    pipeline.safety_checker = None

    # Configure sampler
    sampler_map = {
        "DDIM": DDIMScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
        "DPM2 Karras": KDPM2DiscreteScheduler,
        "DPM2 a Karras": KDPM2AncestralDiscreteScheduler,
    }
    pipeline.scheduler = sampler_map.get(sampler_name, DDIMScheduler).from_config(pipeline.scheduler.config)

    # Device and optimization settings
    if platform.system() == "Darwin":
        pipeline.to("mps" if ia_check_versions.torch_mps_is_available else "cpu")
        pipeline.enable_attention_slicing()
        generator = torch.Generator(devices.cpu)
    else:
        if ia_check_versions.diffusers_enable_cpu_offload and devices.device != devices.cpu:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(devices.device)
        if shared.xformers_available:
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            pipeline.enable_attention_slicing()
        generator = torch.Generator(devices.device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size
    output_images = []

    for count in range(int(iteration_count)):
        gc.collect()
        current_seed = random.randint(0, 2147483647) if seed < 0 or count > 0 else seed
        generator.manual_seed(current_seed)

        pipeline_args = {
            "prompt": prompt,
            "image": init_image,
            "width": width,
            "height": height,
            "mask_image": mask_image,
            "num_inference_steps": sampling_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt,
            "generator": generator,
        }

        output_image = pipeline(**pipeline_args).images[0]

        if composite_enabled:
            dilated_mask = Image.fromarray(
                cv2.dilate(np.array(mask_image), np.ones((3, 3), dtype=np.uint8), iterations=4)
            )
            output_image = Image.composite(
                output_image,
                init_image,
                dilated_mask.convert("L").filter(ImageFilter.GaussianBlur(3))
            )

        # Save with metadata
        params = {
            "Steps": sampling_steps,
            "Sampler": sampler_name,
            "CFG scale": guidance_scale,
            "Seed": current_seed,
            "Size": f"{width}x{height}",
            "Model": model_id,
        }
        infotext = f"{prompt}\nNegative prompt: {negative_prompt}\n{', '.join(f'{k}: {v}' for k, v in params.items())}"
        metadata = PngInfo()
        metadata.add_text("parameters", infotext)
        filename = f"{ia_file_manager.savename_prefix}_{os.path.basename(model_id)}_{current_seed}.png"
        save_path = os.path.join(ia_file_manager.outputs_dir, filename)
        output_image.save(save_path, pnginfo=metadata)

        output_images.append(output_image)
        yield output_images, max(1, iteration_count - (count + 1))


@offload_reload_decorator
@clear_cache_decorator
def run_cleaner(
    input_image: np.ndarray,
    selected_mask: dict,
    model_id: str,
    save_mask_enabled: bool
) -> list:
    """
    Applies a cleaner model to the masked area of the input image.

    Args:
        input_image (np.ndarray): The input image.
        selected_mask (dict): State of the selected mask image editor.
        model_id (str): Identifier of the cleaner model.
        save_mask_enabled (bool): If True, saves the mask image.

    Returns:
        list: List containing the cleaned image.
    """
    global sam_state
    if input_image is None or sam_state["mask_image"] is None or selected_mask is None:
        ia_logging.error("Input image or mask missing")
        return None

    mask_image = sam_state["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("Input image and mask dimensions do not match")
        return None

    save_mask_image(mask_image, save_mask_enabled)

    # Load cleaner model
    device = devices.cpu if platform.system() == "Darwin" else devices.device
    model = ModelManager(name=model_id, device=device)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    # Process image
    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim,
    )
    output_image = model(image=np.array(init_image), mask=np.array(mask_image.convert("L")), config=config)
    output_image = Image.fromarray(cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB))

    # Save output
    filename = f"{ia_file_manager.savename_prefix}_{os.path.basename(model_id)}.png"
    save_path = os.path.join(ia_file_manager.outputs_dir, filename)
    output_image.save(save_path)

    del model
    return [output_image]


@clear_cache_decorator
def get_alpha_channel_image(
    input_image: np.ndarray,
    selected_mask: dict
) -> tuple:
    """
    Creates an image with the mask as its alpha channel.

    Args:
        input_image (np.ndarray): The input image.
        selected_mask (dict): State of the selected mask image editor.

    Returns:
        tuple: PIL image with alpha channel and status message.
    """
    global sam_state
    if input_image is None or sam_state["mask_image"] is None or selected_mask is None:
        ia_logging.error("Input image or mask missing")
        return None, ""

    mask_image = sam_state["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("Input image and mask dimensions do not match")
        return None, ""

    alpha_image = Image.fromarray(input_image).convert("RGBA")
    alpha_image.putalpha(Image.fromarray(mask_image).convert("L"))

    filename = f"{ia_file_manager.savename_prefix}_rgba_image.png"
    save_path = os.path.join(ia_file_manager.outputs_dir, filename)
    alpha_image.save(save_path)

    return alpha_image, f"Saved: {save_path}"


@clear_cache_decorator
def get_mask_image(selected_mask: dict) -> np.ndarray:
    """
    Retrieves and saves the current mask image.

    Args:
        selected_mask (dict): State of the selected mask image editor.

    Returns:
        np.ndarray: The mask image.
    """
    global sam_state
    if sam_state["mask_image"] is None or selected_mask is None:
        return None

    mask_image = sam_state["mask_image"]
    filename = f"{ia_file_manager.savename_prefix}_created_mask.png"
    save_path = os.path.join(ia_file_manager.outputs_dir, filename)
    Image.fromarray(mask_image).save(save_path)

    return mask_image


@clear_cache_decorator
def run_controlnet_inpaint(
    input_image: np.ndarray,
    selected_mask: dict,
    prompt: str,
    negative_prompt: str,
    sampler_id: str,
    sampling_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    module_id: str,
    model_id: str,
    save_mask_enabled: bool,
    low_vram_enabled: bool,
    weight: float,
    mode: str,
    iteration_count: int = 1,
    reference_module_id: str = None,
    reference_image: np.ndarray = None,
    reference_weight: float = 1.0,
    reference_mode: str = "Balanced",
    reference_resize_mode: str = "resize",
    ip_adapter_or_reference: str = None,
    ip_adapter_model_id: str = None
):
    """
    Performs inpainting using ControlNet.

    Args:
        input_image (np.ndarray): The input image.
        selected_mask (dict): State of the selected mask image editor.
        prompt (str): Positive prompt.
        negative_prompt (str): Negative prompt.
        sampler_id (str): Sampler identifier.
        sampling_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale.
        strength (float): Denoising strength.
        seed (int): Random seed (-1 for random).
        module_id (str): ControlNet preprocessor module.
        model_id (str): ControlNet model identifier.
        save_mask_enabled (bool): If True, saves the mask.
        low_vram_enabled (bool): If True, optimizes for low VRAM.
        weight (float): Control weight.
        mode (str): Control mode.
        iteration_count (int, optional): Number of iterations. Defaults to 1.
        reference_module_id (str, optional): Reference module ID. Defaults to None.
        reference_image (np.ndarray, optional): Reference image. Defaults to None.
        reference_weight (float, optional): Reference weight. Defaults to 1.0.
        reference_mode (str, optional): Reference mode. Defaults to "Balanced".
        reference_resize_mode (str, optional): Reference resize mode. Defaults to "resize".
        ip_adapter_or_reference (str, optional): IP-Adapter or Reference-Only choice. Defaults to None.
        ip_adapter_model_id (str, optional): IP-Adapter model ID. Defaults to None.

    Yields:
        tuple: List of output images and remaining iterations.
    """
    global sam_state
    if input_image is None or sam_state["mask_image"] is None or selected_mask is None:
        ia_logging.error("Input image or mask missing")
        return

    mask_image = sam_state["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("Input image and mask dimensions do not match")
        return

    await_pre_reload_model_weights()

    # Compatibility checks
    if shared.sd_model.parameterization == "v" and "sd15" in model_id:
        error_image = draw_text_image(input_image, "SDv2 model incompatible with ControlNet")
        yield [error_image], 1
        return
    if getattr(shared.sd_model, "is_sdxl", False) and "sd15" in model_id:
        error_image = draw_text_image(input_image, "SDXL model incompatible with ControlNet")
        yield [error_image], 1
        return

    controlnet = sam_state.get("controlnet")
    if controlnet is None:
        ia_logging.warning("ControlNet extension not loaded")
        return

    save_mask_image(mask_image, save_mask_enabled)
    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    width, height = init_image.size

    # Prepare processing
    input_mask = None if "inpaint_only" in module_id else mask_image
    processing = get_sd_img2img_processing(
        init_image, input_mask, prompt, negative_prompt, sampler_id, sampling_steps, guidance_scale, strength, seed
    )

    backup_alwayson_scripts(processing.scripts)
    disable_alwayson_scripts_wo_cn(controlnet, processing.scripts)

    # Configure ControlNet units
    control_units = [
        controlnet.to_processing_unit({
            "enabled": True,
            "module": module_id,
            "model": model_id,
            "weight": weight,
            "image": {"image": np.array(init_image), "mask": np.array(mask_image)},
            "resize_mode": controlnet.ResizeMode.RESIZE,
            "low_vram": low_vram_enabled,
            "processor_res": min(width, height),
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "pixel_perfect": True,
            "control_mode": mode,
            "threshold_a": 0.5,
            "threshold_b": 0.5,
        })
    ]

    if reference_module_id and reference_image is not None:
        if reference_resize_mode == "tile":
            ref_height, ref_width = reference_image.shape[:2]
            num_h = math.ceil(height / ref_height) if height > ref_height else 1
            num_h += 1 if num_h % 2 == 0 else 0
            num_w = math.ceil(width / ref_width) if width > ref_width else 1
            num_w += 1 if num_w % 2 == 0 else 0
            reference_image = np.tile(reference_image, (num_h, num_w, 1))
            reference_image = transforms.functional.center_crop(Image.fromarray(reference_image), (height, width))
        else:
            reference_image = ImageOps.fit(
                Image.fromarray(reference_image), (width, height), method=Image.Resampling.LANCZOS
            )

        ref_model_id = "None"
        if ip_adapter_or_reference == "IP-Adapter" and ip_adapter_model_id:
            ipa_modules = [cn for cn in controlnet.get_modules() if "ip-adapter" in cn and "sd15" in cn]
            if ipa_modules:
                reference_module_id = ipa_modules[0]
                ref_model_id = ip_adapter_model_id

        control_units.append(
            controlnet.to_processing_unit({
                "enabled": True,
                "module": reference_module_id,
                "model": ref_model_id,
                "weight": reference_weight,
                "image": {"image": np.array(reference_image), "mask": None},
                "resize_mode": controlnet.ResizeMode.RESIZE,
                "low_vram": low_vram_enabled,
                "processor_res": min(width, height),
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "pixel_perfect": True,
                "control_mode": reference_mode,
                "threshold_a": 0.5,
                "threshold_b": 0.5,
            })
        )

    processing.script_args = np.zeros(get_controlnet_args_to(controlnet, processing.scripts)).tolist()
    controlnet.update_cn_script_in_processing(processing, control_units)

    output_images = []
    clean_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", model_id).strip()

    for count in range(int(iteration_count)):
        gc.collect()
        current_seed = random.randint(0, 2147483647) if seed < 0 or count > 0 else seed
        processing.init_images = [init_image]
        processing.seed = current_seed

        try:
            processed = process_images(processing)
        except devices.NansException:
            error_image = draw_text_image(input_image, "All NaNs produced in VAE")
            clear_controlnet_cache(controlnet, processing.scripts)
            restore_alwayson_scripts(processing.scripts)
            yield [error_image], 1
            return

        if processed and processed.images:
            output_image = processed.images[0]
            infotext = create_infotext(processing, processing.all_prompts, processing.all_seeds, processing.all_subseeds)
            metadata = PngInfo()
            metadata.add_text("parameters", infotext)
            filename = f"{ia_file_manager.savename_prefix}_{os.path.basename(clean_model_id)}_{current_seed}.png"
            save_path = os.path.join(ia_file_manager.outputs_dir, filename)
            output_image.save(save_path, pnginfo=metadata)
            output_images.append(output_image)
            yield output_images, max(1, iteration_count - (count + 1))

    clear_controlnet_cache(controlnet, processing.scripts)
    restore_alwayson_scripts(processing.scripts)


@clear_cache_decorator
def run_webui_inpaint(
    input_image: np.ndarray,
    selected_mask: dict,
    prompt: str,
    negative_prompt: str,
    sampler_id: str,
    sampling_steps: int,
    guidance_scale: float,
    strength: float,
    seed: int,
    model_id: str,
    save_mask_enabled: bool,
    mask_blur: int,
    fill_mode: int,
    iteration_count: int = 1,
    enable_refiner_enabled: bool = False,
    refiner_checkpoint: str = "",
    refiner_switch_at: float = 0.8
):
    """
    Performs inpainting using the web UI's built-in model.

    Args:
        input_image (np.ndarray): The input image.
        selected_mask (dict): State of the selected mask image editor.
        prompt (str): Positive prompt.
        negative_prompt (str): Negative prompt.
        sampler_id (str): Sampler identifier.
        sampling_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale.
        strength (float): Denoising strength.
        seed (int): Random seed (-1 for random).
        model_id (str): Model identifier.
        save_mask_enabled (bool): If True, saves the mask.
        mask_blur (int): Mask blur radius.
        fill_mode (int): Masked content mode (0: fill, 1: original, 2: latent noise, 3: latent nothing).
        iteration_count (int, optional): Number of iterations. Defaults to 1.
        enable_refiner_enabled (bool, optional): If True, enables refiner. Defaults to False.
        refiner_checkpoint (str, optional): Refiner checkpoint. Defaults to "".
        refiner_switch_at (float, optional): Refiner switch point. Defaults to 0.8.

    Yields:
        tuple: List of output images and remaining iterations.
    """
    global sam_state
    if input_image is None or sam_state["mask_image"] is None or selected_mask is None:
        ia_logging.error("Input image or mask missing")
        return

    mask_image = sam_state["mask_image"]
    if input_image.shape != mask_image.shape:
        ia_logging.error("Input image and mask dimensions do not match")
        return

    info = get_closet_checkpoint_match(model_id)
    if info is None:
        ia_logging.error(f"No model found: {model_id}")
        return

    await_backup_reload_ckpt_info(info=info)

    if not getattr(shared.sd_model, "is_sdxl", False) and "sdxl_vae" in getattr(shared.opts, "sd_vae", ""):
        error_image = draw_text_image(input_image, "SDXL VAE incompatible with inpainting model")
        yield [error_image], 1
        return

    set_ia_config(IAConfig.KEYS.INP_WEBUI_MODEL_ID, model_id, IAConfig.SECTIONS.USER)
    save_mask_image(mask_image, save_mask_enabled)

    init_image, mask_image = auto_resize_to_pil(input_image, mask_image)
    processing = get_sd_img2img_processing(
        init_image, mask_image, prompt, negative_prompt, sampler_id, sampling_steps, guidance_scale, strength, seed,
        mask_blur, fill_mode
    )

    backup_alwayson_scripts(processing.scripts)
    disable_all_alwayson_scripts(processing.scripts)
    processing.script_args = np.zeros(get_max_args_to(processing.scripts)).tolist()

    if ia_check_versions.webui_refiner_is_available and enable_refiner_enabled:
        processing.refiner_checkpoint = refiner_checkpoint
        processing.refiner_switch_at = refiner_switch_at

    clean_model_id = re.sub(r"\s\[[0-9a-f]{8,10}\]", "", model_id).strip()
    clean_model_id = os.path.splitext(clean_model_id)[0]
    output_images = []

    for count in range(int(iteration_count)):
        gc.collect()
        current_seed = random.randint(0, 2147483647) if seed < 0 or count > 0 else seed
        processing.init_images = [init_image]
        processing.seed = current_seed

        try:
            processed = process_images(processing)
        except devices.NansException:
            error_image = draw_text_image(input_image, "All NaNs produced in VAE")
            restore_alwayson_scripts(processing.scripts)
            yield [error_image], 1
            return

        if processed and processed.images:
            output_image = processed.images[0]
            infotext = create_infotext(processing, processing.all_prompts, processing.all_seeds, processing.all_subseeds)
            metadata = PngInfo()
            metadata.add_text("parameters", infotext)
            filename = f"{ia_file_manager.savename_prefix}_{os.path.basename(clean_model_id)}_{current_seed}.png"
            save_path = os.path.join(ia_file_manager.outputs_dir, filename)
            output_image.save(save_path, pnginfo=metadata)
            output_images.append(output_image)
            yield output_images, max(1, iteration_count - (count + 1))

    restore_alwayson_scripts(processing.scripts)


def create_ui_tabs():
    """
    Sets up the Gradio interface for the Inpaint Anything extension.

    Returns:
        list: List of tuples defining the UI tab configuration.
    """
    global sam_state
    setup_ia_config_ini()
    sampler_names = get_sampler_names()
    sam_model_ids = get_sam_model_ids()
    sam_model_index = get_ia_config_index(IAConfig.KEYS.SAM_MODEL_ID, IAConfig.SECTIONS.USER)
    inp_model_ids = get_inp_model_ids()
    inp_model_index = get_ia_config_index(IAConfig.KEYS.INP_MODEL_ID, IAConfig.SECTIONS.USER)
    cleaner_model_ids = get_cleaner_model_ids()
    padding_mode_names = get_padding_mode_names()
    sam_state["controlnet"] = find_controlnet()

    # ControlNet setup
    controlnet_enabled = False
    if sam_state["controlnet"]:
        cn_module_ids = [cn for cn in sam_state["controlnet"].get_modules() if "inpaint" in cn]
        cn_module_index = cn_module_ids.index("inpaint_only") if "inpaint_only" in cn_module_ids else 0
        cn_model_ids = [cn for cn in sam_state["controlnet"].get_models() if "inpaint" in cn]
        cn_modes = [mode.value for mode in sam_state["controlnet"].ControlMode]
        controlnet_enabled = len(cn_module_ids) > 0 and len(cn_model_ids) > 0

    cn_sampler_ids = [sampler.name for sampler in samplers_for_img2img] if samplers_for_img2img else ["DDIM"]
    cn_sampler_index = cn_sampler_ids.index("DDIM") if "DDIM" in cn_sampler_ids else 0

    # Reference ControlNet setup
    reference_enabled = False
    if controlnet_enabled and sam_state["controlnet"].get_max_models_num() > 1:
        cn_ref_module_ids = [cn for cn in sam_state["controlnet"].get_modules() if "reference" in cn]
        reference_enabled = len(cn_ref_module_ids) > 0

    ip_adapter_enabled = False
    if reference_enabled:
        cn_ipa_module_ids = [cn for cn in sam_state["controlnet"].get_modules() if "ip-adapter" in cn and "sd15" in cn]
        cn_ipa_model_ids = [cn for cn in sam_state["controlnet"].get_models() if "ip-adapter" in cn and "sd15" in cn]
        ip_adapter_enabled = len(cn_ipa_module_ids) > 0 and len(cn_ipa_model_ids) > 0

    # Web UI inpainting setup
    webui_inpaint_enabled = len(get_inp_webui_model_ids()) > 0
    if webui_inpaint_enabled:
        webui_model_ids = get_inp_webui_model_ids()
        webui_model_index = get_ia_config_index(IAConfig.KEYS.INP_WEBUI_MODEL_ID, IAConfig.SECTIONS.USER)
        webui_sampler_ids = [sampler.name for sampler in samplers_for_img2img] if samplers_for_img2img else ["DDIM"]
        webui_sampler_index = webui_sampler_ids.index("DDIM") if "DDIM" in webui_sampler_ids else 0

    gallery_kwargs = {"columns": 2, "height": 520, "object_fit": "contain", "preview": True}

    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        segment_anything_model_dropdown = gr.Dropdown(
                            label="Segment Anything Model ID",
                            elem_id="sam_model_id",
                            choices=sam_model_ids,
                            value=sam_model_ids[sam_model_index],
                            show_label=True,
                        )
                    with gr.Column():
                        with gr.Row():
                            download_model_button = gr.Button("Download Model", elem_id="load_model_button")
                        with gr.Row():
                            status_textbox = gr.Textbox(
                                label="",
                                elem_id="status_text",
                                max_lines=1,
                                show_label=False,
                                interactive=False,
                            )
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        elem_id="ia_input_image",
                        source="upload",
                        type="numpy",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Accordion("Padding Options", elem_id="padding_options", open=False):
                        with gr.Row():
                            with gr.Column():
                                padding_scale_width_slider = gr.Slider(
                                    label="Scale Width",
                                    elem_id="pad_scale_width",
                                    minimum=1.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.01,
                                )
                            with gr.Column():
                                padding_left_right_balance_slider = gr.Slider(
                                    label="Left/Right Balance",
                                    elem_id="pad_lr_barance",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.01,
                                )
                        with gr.Row():
                            with gr.Column():
                                padding_scale_height_slider = gr.Slider(
                                    label="Scale Height",
                                    elem_id="pad_scale_height",
                                    minimum=1.0,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.01,
                                )
                            with gr.Column():
                                padding_top_bottom_balance_slider = gr.Slider(
                                    label="Top/Bottom Balance",
                                    elem_id="pad_tb_barance",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.5,
                                    step=0.01,
                                )
                        with gr.Row():
                            with gr.Column():
                                padding_mode_dropdown = gr.Dropdown(
                                    label="Padding Mode",
                                    elem_id="padding_mode",
                                    choices=padding_mode_names,
                                    value="edge",
                                )
                            with gr.Column():
                                run_padding_button = gr.Button("Run Padding", elem_id="padding_button")

                with gr.Row():
                    with gr.Column():
                        anime_style_checkbox = gr.Checkbox(
                            label="Anime Style (Up Detection, Down Mask Quality)",
                            elem_id="anime_style_checkbox",
                            show_label=True,
                            interactive=True,
                        )
                    with gr.Column():
                        run_segment_anything_button = gr.Button(
                            "Run Segment Anything",
                            elem_id="sam_button",
                            variant="primary",
                            interactive=False,
                        )

                with gr.Tab("Inpainting", elem_id="inpainting_tab"):
                    with gr.Row():
                        with gr.Column():
                            inpainting_prompt_textbox = gr.Textbox(
                                label="Inpainting Prompt",
                                elem_id="ia_sd_prompt",
                            )
                            inpainting_negative_prompt_textbox = gr.Textbox(
                                label="Negative Prompt",
                                elem_id="ia_sd_n_prompt",
                            )
                        with gr.Column(scale=0, min_width=128):
                            gr.Markdown("Get prompt from:")
                            get_txt2img_prompt_button = gr.Button("txt2img", elem_id="get_txt2img_prompt_button")
                            get_img2img_prompt_button = gr.Button("img2img", elem_id="get_img2img_prompt_button")
                    with gr.Accordion("Advanced Options", elem_id="inp_advanced_options", open=False):
                        composite_checkbox = gr.Checkbox(
                            label="Mask Area Only",
                            elem_id="composite_checkbox",
                            value=True,
                            show_label=True,
                            interactive=True,
                        )
                        with gr.Row():
                            with gr.Column():
                                sampler_dropdown = gr.Dropdown(
                                    label="Sampler",
                                    elem_id="sampler_name",
                                    choices=sampler_names,
                                    value=sampler_names[0],
                                    show_label=True,
                                )
                            with gr.Column():
                                sampling_steps_slider = gr.Slider(
                                    label="Sampling Steps",
                                    elem_id="ddim_steps",
                                    minimum=1,
                                    maximum=100,
                                    value=20,
                                    step=1,
                                )
                        guidance_scale_slider = gr.Slider(
                            label="Guidance Scale",
                            elem_id="cfg_scale",
                            minimum=0.1,
                            maximum=30.0,
                            value=7.5,
                            step=0.1,
                        )
                        seed_slider = gr.Slider(
                            label="Seed",
                            elem_id="sd_seed",
                            minimum=-1,
                            maximum=2147483647,
                            step=1,
                            value=-1,
                        )
                    with gr.Row():
                        with gr.Column():
                            inpainting_model_dropdown = gr.Dropdown(
                                label="Inpainting Model ID",
                                elem_id="inp_model_id",
                                choices=inp_model_ids,
                                value=inp_model_ids[inp_model_index],
                                show_label=True,
                            )
                        with gr.Column():
                            with gr.Row():
                                run_inpaint_button = gr.Button("Run Inpainting", elem_id="inpaint_button", variant="primary")
                            with gr.Row():
                                save_mask_checkbox = gr.Checkbox(
                                    label="Save Mask",
                                    elem_id="save_mask_checkbox",
                                    value=False,
                                    show_label=False,
                                    interactive=False,
                                    visible=False,
                                )
                                iteration_count_slider = gr.Slider(
                                    label="Iterations",
                                    elem_id="iteration_count",
                                    minimum=1,
                                    maximum=10,
                                    value=1,
                                    step=1,
                                )
                    with gr.Row():
                        inpainting_output_gallery = gr.Gallery(
                            label="Inpainted Image",
                            elem_id="ia_out_image",
                            show_label=False,
                            **gallery_kwargs,
                        )

                with gr.Tab("Cleaner", elem_id="cleaner_tab"):
                    with gr.Row():
                        with gr.Column():
                            cleaner_model_dropdown = gr.Dropdown(
                                label="Cleaner Model ID",
                                elem_id="cleaner_model_id",
                                choices=cleaner_model_ids,
                                value=cleaner_model_ids[0],
                                show_label=True,
                            )
                        with gr.Column():
                            with gr.Row():
                                run_cleaner_button = gr.Button("Run Cleaner", elem_id="cleaner_button", variant="primary")
                            with gr.Row():
                                cleaner_save_mask_checkbox = gr.Checkbox(
                                    label="Save Mask",
                                    elem_id="cleaner_save_mask_checkbox",
                                    value=False,
                                    show_label=False,
                                    interactive=False,
                                    visible=False,
                                )
                    with gr.Row():
                        cleaner_output_gallery = gr.Gallery(
                            label="Cleaned Image",
                            elem_id="ia_cleaner_out_image",
                            show_label=False,
                            **gallery_kwargs,
                        )

                if webui_inpaint_enabled:
                    with gr.Tab("Inpainting Web UI", elem_id="webui_inpainting_tab"):
                        with gr.Row():
                            with gr.Column():
                                webui_prompt_textbox = gr.Textbox(
                                    label="Inpainting Prompt",
                                    elem_id="ia_webui_sd_prompt",
                                )
                                webui_negative_prompt_textbox = gr.Textbox(
                                    label="Negative Prompt",
                                    elem_id="ia_webui_sd_n_prompt",
                                )
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                webui_get_txt2img_prompt_button = gr.Button(
                                    "txt2img",
                                    elem_id="webui_get_txt2img_prompt_button",
                                )
                                webui_get_img2img_prompt_button = gr.Button(
                                    "img2img",
                                    elem_id="webui_get_img2img_prompt_button",
                                )
                        with gr.Accordion("Advanced Options", elem_id="webui_advanced_options", open=False):
                            webui_mask_blur_slider = gr.Slider(
                                label="Mask Blur",
                                minimum=0,
                                maximum=64,
                                step=1,
                                value=4,
                                elem_id="webui_mask_blur",
                            )
                            webui_fill_mode_radio = gr.Radio(
                                label="Masked Content",
                                elem_id="webui_fill_mode",
                                choices=["fill", "original", "latent noise", "latent nothing"],
                                value="original",
                                type="index",
                            )
                            with gr.Row():
                                with gr.Column():
                                    webui_sampler_dropdown = gr.Dropdown(
                                        label="Sampling Method Web UI",
                                        elem_id="webui_sampler_id",
                                        choices=webui_sampler_ids,
                                        value=webui_sampler_ids[webui_sampler_index],
                                        show_label=True,
                                    )
                                with gr.Column():
                                    webui_sampling_steps_slider = gr.Slider(
                                        label="Sampling Steps Web UI",
                                        elem_id="webui_ddim_steps",
                                        minimum=1,
                                        maximum=150,
                                        value=30,
                                        step=1,
                                    )
                            webui_guidance_scale_slider = gr.Slider(
                                label="Guidance Scale Web UI",
                                elem_id="webui_cfg_scale",
                                minimum=0.1,
                                maximum=30.0,
                                value=7.5,
                                step=0.1,
                            )
                            webui_strength_slider = gr.Slider(
                                label="Denoising Strength Web UI",
                                elem_id="webui_strength",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.75,
                                step=0.01,
                            )
                            webui_seed_slider = gr.Slider(
                                label="Seed",
                                elem_id="webui_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        if ia_check_versions.webui_refiner_is_available:
                            with gr.Accordion("Refiner Options", elem_id="webui_refiner_options", open=False):
                                with gr.Row():
                                    webui_enable_refiner_checkbox = gr.Checkbox(
                                        label="Enable Refiner",
                                        elem_id="webui_enable_refiner_checkbox",
                                        value=False,
                                        show_label=True,
                                        interactive=True,
                                    )
                                with gr.Row():
                                    webui_refiner_checkpoint_dropdown = gr.Dropdown(
                                        label="Refiner Model ID",
                                        elem_id="webui_refiner_checkpoint",
                                        choices=shared.list_checkpoint_tiles(),
                                        value="",
                                    )
                                    webui_refiner_switch_at_slider = gr.Slider(
                                        value=0.8,
                                        label="Switch At",
                                        minimum=0.01,
                                        maximum=1.0,
                                        step=0.01,
                                        elem_id="webui_refiner_switch_at",
                                    )
                        with gr.Row():
                            with gr.Column():
                                webui_model_dropdown = gr.Dropdown(
                                    label="Inpainting Model ID Web UI",
                                    elem_id="webui_model_id",
                                    choices=webui_model_ids,
                                    value=webui_model_ids[webui_model_index],
                                    show_label=True,
                                )
                            with gr.Column():
                                with gr.Row():
                                    webui_run_inpaint_button = gr.Button(
                                        "Run Inpainting",
                                        elem_id="webui_inpaint_button",
                                        variant="primary",
                                    )
                                with gr.Row():
                                    webui_save_mask_checkbox = gr.Checkbox(
                                        label="Save Mask",
                                        elem_id="webui_save_mask_checkbox",
                                        value=False,
                                        show_label=False,
                                        interactive=False,
                                        visible=False,
                                    )
                                    webui_iteration_count_slider = gr.Slider(
                                        label="Iterations",
                                        elem_id="webui_iteration_count",
                                        minimum=1,
                                        maximum=10,
                                        value=1,
                                        step=1,
                                    )
                        with gr.Row():
                            webui_output_gallery = gr.Gallery(
                                label="Inpainted Image",
                                elem_id="ia_webui_out_image",
                                show_label=False,
                                **gallery_kwargs,
                            )

                with gr.Tab("ControlNet Inpaint", elem_id="cn_inpaint_tab"):
                    if controlnet_enabled:
                        with gr.Row():
                            with gr.Column():
                                controlnet_prompt_textbox = gr.Textbox(
                                    label="Inpainting Prompt",
                                    elem_id="ia_cn_sd_prompt",
                                )
                                controlnet_negative_prompt_textbox = gr.Textbox(
                                    label="Negative Prompt",
                                    elem_id="ia_cn_sd_n_prompt",
                                )
                            with gr.Column(scale=0, min_width=128):
                                gr.Markdown("Get prompt from:")
                                controlnet_get_txt2img_prompt_button = gr.Button(
                                    "txt2img",
                                    elem_id="cn_get_txt2img_prompt_button",
                                )
                                controlnet_get_img2img_prompt_button = gr.Button(
                                    "img2img",
                                    elem_id="cn_get_img2img_prompt_button",
                                )
                        with gr.Accordion("Advanced Options", elem_id="cn_advanced_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    controlnet_sampler_dropdown = gr.Dropdown(
                                        label="Sampling Method",
                                        elem_id="cn_sampler_id",
                                        choices=cn_sampler_ids,
                                        value=cn_sampler_ids[cn_sampler_index],
                                        show_label=True,
                                    )
                                with gr.Column():
                                    controlnet_sampling_steps_slider = gr.Slider(
                                        label="Sampling Steps",
                                        elem_id="cn_ddim_steps",
                                        minimum=1,
                                        maximum=150,
                                        value=30,
                                        step=1,
                                    )
                            controlnet_guidance_scale_slider = gr.Slider(
                                label="Guidance Scale",
                                elem_id="cn_cfg_scale",
                                minimum=0.1,
                                maximum=30.0,
                                value=7.5,
                                step=0.1,
                            )
                            controlnet_strength_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                label="Denoising Strength",
                                value=0.75,
                                elem_id="cn_strength",
                            )
                            controlnet_seed_slider = gr.Slider(
                                label="Seed",
                                elem_id="cn_sd_seed",
                                minimum=-1,
                                maximum=2147483647,
                                step=1,
                                value=-1,
                            )
                        with gr.Accordion("ControlNet Options", elem_id="cn_cn_options", open=False):
                            with gr.Row():
                                with gr.Column():
                                    controlnet_low_vram_checkbox = gr.Checkbox(
                                        label="Low VRAM",
                                        elem_id="cn_low_vram_checkbox",
                                        value=True,
                                        show_label=True,
                                        interactive=True,
                                    )
                                    controlnet_weight_slider = gr.Slider(
                                        label="Control Weight",
                                        elem_id="cn_weight",
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.05,
                                    )
                                with gr.Column():
                                    controlnet_mode_dropdown = gr.Dropdown(
                                        label="Control Mode",
                                        elem_id="cn_mode",
                                        choices=cn_modes,
                                        value=cn_modes[-1],
                                        show_label=True,
                                    )
                            if reference_enabled:
                                with gr.Row():
                                    with gr.Column():
                                        markdown_text = "Reference Control (enabled with image below)"
                                        if not ip_adapter_enabled:
                                            markdown_text += (
                                                "<br><span style='color: gray;'>"
                                                "[IP-Adapter](https://huggingface.co/lllyasviel/sd_control_collection/tree/main) "
                                                "is not available. Reference-Only is used.</span>"
                                            )
                                        gr.Markdown(markdown_text)
                                        if ip_adapter_enabled:
                                            controlnet_ip_adapter_or_reference_radio = gr.Radio(
                                                label="IP-Adapter or Reference-Only",
                                                elem_id="cn_ipa_or_ref",
                                                choices=["IP-Adapter", "Reference-Only"],
                                                value="IP-Adapter",
                                                show_label=False,
                                            )
                                        controlnet_reference_image = gr.Image(
                                            label="Reference Image",
                                            elem_id="cn_ref_image",
                                            source="upload",
                                            type="numpy",
                                            interactive=True,
                                        )
                                    with gr.Column():
                                        controlnet_reference_resize_mode_radio = gr.Radio(
                                            label="Reference Image Resize Mode",
                                            elem_id="cn_ref_resize_mode",
                                            choices=["resize", "tile"],
                                            value="resize",
                                            show_label=True,
                                        )
                                        if ip_adapter_enabled:
                                            controlnet_ip_adapter_model_dropdown = gr.Dropdown(
                                                label="IP-Adapter Model ID",
                                                elem_id="cn_ipa_model_id",
                                                choices=cn_ipa_model_ids,
                                                value=cn_ipa_model_ids[0],
                                                show_label=True,
                                            )
                                        controlnet_reference_module_dropdown = gr.Dropdown(
                                            label="Reference Type for Reference-Only",
                                            elem_id="cn_ref_module_id",
                                            choices=cn_ref_module_ids,
                                            value=cn_ref_module_ids[-1],
                                            show_label=True,
                                        )
                                        controlnet_reference_weight_slider = gr.Slider(
                                            label="Reference Control Weight",
                                            elem_id="cn_ref_weight",
                                            minimum=0.0,
                                            maximum=2.0,
                                            value=1.0,
                                            step=0.05,
                                        )
                                        controlnet_reference_mode_dropdown = gr.Dropdown(
                                            label="Reference Control Mode",
                                            elem_id="cn_ref_mode",
                                            choices=cn_modes,
                                            value=cn_modes[0],
                                            show_label=True,
                                        )
                            else:
                                gr.Markdown(
                                    "The Multi ControlNet setting is currently set to 1.<br>"
                                    "If you wish to use the Reference-Only Control, "
                                    "please adjust the Multi ControlNet setting to 2 or more and restart the Web UI."
                                )
                        with gr.Row():
                            with gr.Column():
                                controlnet_module_dropdown = gr.Dropdown(
                                    label="ControlNet Preprocessor",
                                    elem_id="cn_module_id",
                                    choices=cn_module_ids,
                                    value=cn_module_ids[cn_module_index],
                                    show_label=True,
                                )
                                controlnet_model_dropdown = gr.Dropdown(
                                    label="ControlNet Model ID",
                                    elem_id="cn_model_id",
                                    choices=cn_model_ids,
                                    value=cn_model_ids[0],
                                    show_label=True,
                                )
                            with gr.Column():
                                with gr.Row():
                                    controlnet_run_inpaint_button = gr.Button(
                                        "Run ControlNet Inpaint",
                                        elem_id="cn_inpaint_button",
                                        variant="primary",
                                    )
                                with gr.Row():
                                    controlnet_save_mask_checkbox = gr.Checkbox(
                                        label="Save Mask",
                                        elem_id="cn_save_mask_checkbox",
                                        value=False,
                                        show_label=False,
                                        interactive=False,
                                        visible=False,
                                    )
                                    controlnet_iteration_count_slider = gr.Slider(
                                        label="Iterations",
                                        elem_id="cn_iteration_count",
                                        minimum=1,
                                        maximum=10,
                                        value=1,
                                        step=1,
                                    )
                        with gr.Row():
                            controlnet_output_gallery = gr.Gallery(
                                label="Inpainted Image",
                                elem_id="ia_cn_out_image",
                                show_label=False,
                                **gallery_kwargs,
                            )
                    else:
                        if sam_state["controlnet"] is None:
                            gr.Markdown(
                                "ControlNet extension is not available.<br>"
                                "Requires the [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension."
                            )
                        elif len(cn_module_ids) > 0:
                            cn_models_dir = os.path.join("extensions", "sd-webui-controlnet", "models")
                            gr.Markdown(
                                "ControlNet inpaint model is not available.<br>"
                                "Requires the [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) inpaint model "
                                f"in the {cn_models_dir} directory."
                            )
                        else:
                            gr.Markdown(
                                "ControlNet inpaint preprocessor is not available.<br>"
                                "The local version of [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension may be old."
                            )

                with gr.Tab("Mask Only", elem_id="mask_only_tab"):
                    with gr.Row():
                        with gr.Column():
                            get_alpha_image_button = gr.Button(
                                "Get Mask as Alpha of Image",
                                elem_id="get_alpha_image_button",
                            )
                        with gr.Column():
                            get_mask_button = gr.Button("Get Mask", elem_id="get_mask_button")
                    with gr.Row():
                        with gr.Column():
                            alpha_output_image = gr.Image(
                                label="Alpha Channel Image",
                                elem_id="alpha_out_image",
                                type="pil",
                                image_mode="RGBA",
                                interactive=False,
                            )
                        with gr.Column():
                            mask_output_image = gr.Image(
                                label="Mask Image",
                                elem_id="mask_out_image",
                                type="numpy",
                                interactive=False,
                            )
                    with gr.Row():
                        with gr.Column():
                            get_alpha_status_textbox = gr.Textbox(
                                label="",
                                elem_id="get_alpha_status_text",
                                max_lines=1,
                                show_label=False,
                                interactive=False,
                            )
                        with gr.Column():
                            send_to_inpaint_button = gr.Button(
                                "Send to Img2Img Inpaint",
                                elem_id="mask_send_to_inpaint_button",
                            )

            with gr.Column():
                with gr.Row():
                    gr.Markdown("Mouse over image: Press `S` key for Fullscreen mode, `R` key to Reset zoom")
                with gr.Row():
                    segment_anything_brush = Brush(default_size=8, default_color="black", colors=["black", "white"])
                    segment_anything_image = ImageEditor(
                        label="Segment Anything Image",
                        elem_id="ia_sam_image",
                        type="numpy",
                        brush=segment_anything_brush,
                        
                    )
                with gr.Row():
                    with gr.Column():
                        create_mask_button = gr.Button("Create Mask", elem_id="select_button", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            invert_mask_checkbox = gr.Checkbox(
                                label="Invert Mask",
                                elem_id="invert_checkbox",
                                show_label=True,
                                interactive=True,
                            )
                            ignore_black_checkbox = gr.Checkbox(
                                label="Ignore Black Area",
                                elem_id="ignore_black_checkbox",
                                value=True,
                                show_label=True,
                                interactive=True,
                            )
                with gr.Row():
                    selected_mask_brush = Brush(default_size=12, default_color="black", colors=["black", "white"])
                    selected_mask_image = ImageEditor(
                        label="Selected Mask Image",
                        elem_id="ia_sel_mask",
                        type="numpy",
                        brush=selected_mask_brush,
                        
                    )
                with gr.Row():
                    with gr.Column():
                        expand_mask_button = gr.Button("Expand Mask Region", elem_id="expand_mask_button")
                        expand_mask_iteration_slider = gr.Slider(
                            label="Expand Mask Iterations",
                            elem_id="expand_mask_iteration_count",
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                        )
                    with gr.Column():
                        trim_mask_button = gr.Button("Trim Mask by Sketch", elem_id="apply_mask_button")
                        add_mask_button = gr.Button("Add Mask by Sketch", elem_id="add_mask_button")

            # Event bindings
            download_model_button.click(
                fn=download_segment_anything_model,
                inputs=[segment_anything_model_dropdown],
                outputs=[status_textbox],
            )
            input_image.upload(
                fn=handle_input_image_upload,
                inputs=[input_image, segment_anything_image, selected_mask_image],
                outputs=[segment_anything_image, selected_mask_image, run_segment_anything_button],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_initSamSelMask")
            run_padding_button.click(
                fn=apply_padding,
                inputs=[
                    input_image,
                    padding_scale_width_slider,
                    padding_scale_height_slider,
                    padding_left_right_balance_slider,
                    padding_top_bottom_balance_slider,
                    padding_mode_dropdown,
                ],
                outputs=[input_image, status_textbox],
            )
            run_segment_anything_button.click(
                fn=run_segment_anything,
                inputs=[input_image, segment_anything_model_dropdown, segment_anything_image, anime_style_checkbox],
                outputs=[segment_anything_image, status_textbox],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSamMask")
            create_mask_button.click(
                fn=select_mask,
                inputs=[
                    input_image,
                    segment_anything_image,
                    invert_mask_checkbox,
                    ignore_black_checkbox,
                    selected_mask_image,
                ],
                outputs=[selected_mask_image],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            expand_mask_button.click(
                fn=expand_mask,
                inputs=[input_image, selected_mask_image, expand_mask_iteration_slider],
                outputs=[selected_mask_image],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            trim_mask_button.click(
                fn=trim_mask_by_sketch,
                inputs=[input_image, selected_mask_image],
                outputs=[selected_mask_image],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            add_mask_button.click(
                fn=add_mask_by_sketch,
                inputs=[input_image, selected_mask_image],
                outputs=[selected_mask_image],
            ).then(fn=None, inputs=None, outputs=None, _js="inpaintAnything_clearSelMask")
            get_txt2img_prompt_button.click(fn=None, inputs=None, outputs=None, _js="inpaintAnything_getTxt2imgPrompt")
            get_img2img_prompt_button.click(fn=None, inputs=None, outputs=None, _js="inpaintAnything_getImg2imgPrompt")
            run_inpaint_button.click(
                fn=run_inpaint,
                inputs=[
                    input_image,
                    selected_mask_image,
                    inpainting_prompt_textbox,
                    inpainting_negative_prompt_textbox,
                    sampling_steps_slider,
                    guidance_scale_slider,
                    seed_slider,
                    inpainting_model_dropdown,
                    save_mask_checkbox,
                    composite_checkbox,
                    sampler_dropdown,
                    iteration_count_slider,
                ],
                outputs=[inpainting_output_gallery, iteration_count_slider],
            )
            run_cleaner_button.click(
                fn=run_cleaner,
                inputs=[input_image, selected_mask_image, cleaner_model_dropdown, cleaner_save_mask_checkbox],
                outputs=[cleaner_output_gallery],
            )
            get_alpha_image_button.click(
                fn=get_alpha_channel_image,
                inputs=[input_image, selected_mask_image],
                outputs=[alpha_output_image, get_alpha_status_textbox],
            )
            get_mask_button.click(
                fn=get_mask_image,
                inputs=[selected_mask_image],
                outputs=[mask_output_image],
            )
            send_to_inpaint_button.click(fn=None, _js="inpaintAnything_sendToInpaint", inputs=None, outputs=None)

            if controlnet_enabled:
                controlnet_get_txt2img_prompt_button.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetTxt2imgPrompt"
                )
                controlnet_get_img2img_prompt_button.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_cnGetImg2imgPrompt"
                )
                cn_inputs = [
                    input_image,
                    selected_mask_image,
                    controlnet_prompt_textbox,
                    controlnet_negative_prompt_textbox,
                    controlnet_sampler_dropdown,
                    controlnet_sampling_steps_slider,
                    controlnet_guidance_scale_slider,
                    controlnet_strength_slider,
                    controlnet_seed_slider,
                    controlnet_module_dropdown,
                    controlnet_model_dropdown,
                    controlnet_save_mask_checkbox,
                    controlnet_low_vram_checkbox,
                    controlnet_weight_slider,
                    controlnet_mode_dropdown,
                    controlnet_iteration_count_slider,
                ]
                if reference_enabled:
                    cn_inputs.extend([
                        controlnet_reference_module_dropdown,
                        controlnet_reference_image,
                        controlnet_reference_weight_slider,
                        controlnet_reference_mode_dropdown,
                        controlnet_reference_resize_mode_radio,
                    ])
                if ip_adapter_enabled:
                    cn_inputs.extend([controlnet_ip_adapter_or_reference_radio, controlnet_ip_adapter_model_dropdown])
                controlnet_run_inpaint_button.click(
                    fn=run_controlnet_inpaint,
                    inputs=cn_inputs,
                    outputs=[controlnet_output_gallery, controlnet_iteration_count_slider],
                ).then(fn=async_post_reload_model_weights, inputs=None, outputs=None)

            if webui_inpaint_enabled:
                webui_get_txt2img_prompt_button.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetTxt2imgPrompt"
                )
                webui_get_img2img_prompt_button.click(
                    fn=None, inputs=None, outputs=None, _js="inpaintAnything_webuiGetImg2imgPrompt"
                )
                webui_inputs = [
                    input_image,
                    selected_mask_image,
                    webui_prompt_textbox,
                    webui_negative_prompt_textbox,
                    webui_sampler_dropdown,
                    webui_sampling_steps_slider,
                    webui_guidance_scale_slider,
                    webui_strength_slider,
                    webui_seed_slider,
                    webui_model_dropdown,
                    webui_save_mask_checkbox,
                    webui_mask_blur_slider,
                    webui_fill_mode_radio,
                    webui_iteration_count_slider,
                ]
                if ia_check_versions.webui_refiner_is_available:
                    webui_inputs.extend([
                        webui_enable_refiner_checkbox,
                        webui_refiner_checkpoint_dropdown,
                        webui_refiner_switch_at_slider,
                    ])
                webui_run_inpaint_button.click(
                    fn=run_webui_inpaint,
                    inputs=webui_inputs,
                    outputs=[webui_output_gallery, webui_iteration_count_slider],
                ).then(fn=async_post_reload_model_weights, inputs=None, outputs=None)

    return [(interface, "Inpaint Anything", "inpaint_anything")]


def configure_ui_settings():
    """
    Adds settings options to the Stable Diffusion web UI for the Inpaint Anything extension.
    """
    section = ("inpaint_anything", "Inpaint Anything")
    shared.opts.add_option(
        "inpaint_anything_save_folder",
        shared.OptionInfo(
            default="inpaint-anything",
            label="Folder name where output images will be saved",
            component=gr.Radio,
            component_args={"choices": ["inpaint-anything", "img2img-images (img2img output setting of web UI)"]},
            section=section,
        ),
    )
    shared.opts.add_option(
        "inpaint_anything_sam_oncpu",
        shared.OptionInfo(
            default=False,
            label="Run Segment Anything on CPU",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "inpaint_anything_offline_inpainting",
        shared.OptionInfo(
            default=False,
            label="Run Inpainting on offline network (Models not auto-downloaded)",
            component=gr.Checkbox,
            component_args={"interactive": True},
            section=section,
        ),
    )
    shared.opts.add_option(
        "inpaint_anything_padding_fill",
        shared.OptionInfo(
            default=127,
            label="Fill value used when Padding is set to constant",
            component=gr.Slider,
            component_args={"minimum": 0, "maximum": 255, "step": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "inpain_anything_sam_models_dir",
        shared.OptionInfo(
            default="",
            label="Segment Anything Models Directory; If empty, defaults to [Inpaint Anything extension folder]/models",
            component=gr.Textbox,
            component_args={"interactive": True},
            section=section,
        ),
    )


# Register callbacks for UI integration with Stable Diffusion web UI
script_callbacks.on_ui_settings(configure_ui_settings)
script_callbacks.on_ui_tabs(create_ui_tabs)