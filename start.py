import argparse
from copy import deepcopy
import re
import os

import bleach
import cv2
# import gradio as gr
from PIL import Image
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from LLaVA.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
# from visual_search import parse_args, VSM, visual_search
# from vstar_bench_eval import normalize_bbox, expand2square, VQA_LLM

from LLaVA.llava.mm_utils import get_model_name_from_path, tokenizer_image_object_token, KeywordsStoppingCriteria
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init


class VQA_LLM:
	def __init__(self, args):
		disable_torch_init()
		model_path = args.vqa_model_path
		model_name = get_model_name_from_path(model_path)
		model_name += 'llava'
		model_base = None
		device_map = "auto"
		self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path,
																								   model_base,
																								   model_name)
		self.conv_type = args.conv_type

	def get_patch(self, bbox, image_width, image_height, patch_size=224, patch_scale=None):
		object_width = int(np.ceil(bbox[2]))
		object_height = int(np.ceil(bbox[3]))

		object_center_x = int(bbox[0] + bbox[2] / 2)
		object_center_y = int(bbox[1] + bbox[3] / 2)

		if patch_scale is None:
			patch_width = max(object_width, patch_size)
			patch_height = max(object_height, patch_size)
		else:
			patch_width = int(object_width * patch_scale)
			patch_height = int(object_height * patch_scale)

		left = max(0, object_center_x - patch_width // 2)
		right = min(left + patch_width, image_width)

		top = max(0, object_center_y - patch_height // 2)
		bottom = min(top + patch_height, image_height)

		return [left, top, right, bottom]

	def get_object_crop(self, image, bbox, patch_scale):
		resized_bbox = self.get_patch(bbox, image.width, image.height, patch_scale=patch_scale)
		object_crop = image.crop((resized_bbox[0], resized_bbox[1], resized_bbox[2], resized_bbox[3]))
		object_crop = object_crop.resize(
			(self.image_processor.crop_size['width'], self.image_processor.crop_size['height']))
		object_crop = self.image_processor.preprocess(object_crop, return_tensors='pt')['pixel_values'][0]
		return object_crop

	@torch.inference_mode()
	def free_form_inference(self, image, question, temperature=0, top_p=None, num_beams=1, max_new_tokens=200,
							object_crops=None, images_long=None, objects_long=None):
		conv = conv_templates[self.conv_type].copy()
		qs = DEFAULT_IMAGE_TOKEN + '\n' + question
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()
		stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
												 return_tensors='pt').unsqueeze(0).cuda()
		image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
		stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

		output_ids = self.model.generate(
			input_ids,
			images=image_tensor.unsqueeze(0).half().cuda(),
			object_features=object_crops.half().cuda() if object_crops is not None else None,
			images_long=images_long,
			objects_long=objects_long,
			do_sample=True if temperature > 0 else False,
			num_beams=num_beams,
			temperature=temperature,
			top_p=top_p,
			max_new_tokens=max_new_tokens,
			use_cache=True,
			stopping_criteria=[stopping_criteria])

		input_token_len = input_ids.shape[1]
		n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
		if n_diff_input_output > 0:
			print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
		outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
		outputs = outputs.strip()
		if outputs.endswith(stop_str):
			outputs = outputs[:-len(stop_str)]
		outputs = outputs.strip()
		return outputs

	@torch.inference_mode()
	def multiple_choices_inference(self, image, question, options, object_crops=None, images_long=None,
								   objects_long=None):
		conv = conv_templates[self.conv_type].copy()
		qs = DEFAULT_IMAGE_TOKEN + '\n' + question
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()

		question_input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
														  return_tensors='pt').unsqueeze(0).cuda()
		image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

		output_question = self.model(
			question_input_ids,
			use_cache=True,
			images=image_tensor.unsqueeze(0).half().cuda(),
			object_features=object_crops.half().cuda() if object_crops is not None else None,
			images_long=images_long,
			objects_long=objects_long)

		question_logits = output_question.logits
		question_past_key_values = output_question.past_key_values

		loss_list = []

		for option in options:
			conv = conv_templates[self.conv_type].copy()
			conv.append_message(conv.roles[0], qs)
			conv.append_message(conv.roles[1], option)
			full_prompt = conv.get_prompt()

			full_input_ids = tokenizer_image_object_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
														  return_tensors='pt').unsqueeze(0).cuda()
			option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

			output_option = self.model(input_ids=option_answer_input_ids,
									   use_cache=True,
									   attention_mask=torch.ones(1, question_logits.shape[1] +
																 option_answer_input_ids.shape[1],
																 device=full_input_ids.device),
									   past_key_values=question_past_key_values)

			logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

			loss_fct = CrossEntropyLoss()
			logits = logits.view(-1, self.model.config.vocab_size)
			labels = option_answer_input_ids.view(-1)
			loss = loss_fct(logits, labels)

			loss_list.append(loss)

		option_chosen = torch.stack(loss_list).argmin()

		return option_chosen.cpu().item()


def parse_args_vqallm(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
	parser.add_argument("--vqa-model-base", type=str, default=None)
	parser.add_argument("--conv_type", default="v1", type=str,)
	parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")  # Visual Semantic Matching - is a technique used in computer vision and natural language processing to match visual content (like images) with semantic information (like text descriptions)
	parser.add_argument("--minimum_size_scale", default=4.0, type=float)
	parser.add_argument("--minimum_size", default=224, type=int)
	return parser.parse_args(args)


# This function converts the input image to a square image
def expand2square(pil_img, background_color):
	width, height = pil_img.size
	if width == height:
		return pil_img, 0, 0
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))
		return result, 0, (width - height) // 2
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))
		return result, (height - width) // 2, 0


example = ["Based on the exact content of the flag on the roof, what can we know about its owner?", "./image_examples/flag.JPG"]

args = parse_args_vqallm({})
# init VQA LLM
vqa = VQA_LLM(args)

missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
focus_msg = "Additional visual information to focus on: "


def main():
	input_str, input_img = example[0], example[1]
	print("input_str: ", input_str, "input_image: ", input_img)

	image = Image.open(input_img).convert('RGB')
	image, _, _ = expand2square(image, tuple(int(x * 255) for x in vqa.image_processor.image_mean))
	prediction = vqa.free_form_inference(image, input_str, max_new_tokens=512)
	print(prediction)
	pass


if __name__ == "__main__":
	main()