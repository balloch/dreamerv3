import time

import cloudpickle
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .. import distr
from .MobileVLM_main.mobilevlm.model.mobilevlm import load_pretrained_model
from .MobileVLM_main.mobilevlm.conversation import conv_templates, SeparatorStyle
from .MobileVLM_main.mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from .MobileVLM_main.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN



class Driver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    model_path = "mtgv/MobileVLM-1.7B" # MobileVLM V2
    model_name =  model_path.split('/')[-1]
    self.model_name = model_name
    image_file = "assets/samples/IntoFire.png"
    prompt_str = "Is there fire?"
    print('Model Path: ',model_path)
    #Activate to use model.
    self.use_vlm = False
    print(f'VLM in use: {self.use_vlm}')
    if self.use_vlm:  
      tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, False, False)
      self.VLM_model = model
      self.VLM_tokenizer = tokenizer
      self.image_processor = image_processor
      self.context_len = context_len
      self.prompt = prompt_str
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          distr.StoppableProcess(self._env_server, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.stop() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode, mode='Train'):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
    ###VLM####
    self.VLM_response(obs[-1]['img'])
    #time.sleep()
    ###VLM####
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}

    assert all(len(x) == self.length for x in obs.values()), obs
    acts, outs, self.carry = policy(obs, self.carry, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self.acts = acts
    trans = {**obs, **acts, **outs}
    for i in range(self.length):
      trn = {k: v[i] for k, v in trans.items()}
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(context, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while context.running:
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        else:
          raise ValueError(f'Invalid message {msg}')
    except Exception as e:
      distr.warn_remote_error(e, f'Env{envid}')
      pipe.send(('error', e))
    finally:
      print(f'Closing env {envid}')
      env.close()
      pipe.close()
  def VLM_response(self, obs):

    if self.use_vlm == False:
      return

    #tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)
    conversation_mode = "v1"
    temperature = 0.2
    top_p = None
    num_beams = 1
    max_new_tokens = 512


    images = [Image.fromarray(np.clip(255 * obs + .5, 0, 255).astype(np.uint8))]
    
    images_tensor = process_images(images, self.image_processor, self.VLM_model.config).to(self.VLM_model.device, dtype=torch.float16)

    conv = conv_templates[conversation_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + self.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, self.VLM_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], self.VLM_tokenizer, input_ids)
    # Inference
    with torch.inference_mode():
        output_ids = self.VLM_model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = self.VLM_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    print(f"🚀 {self.model_name}: {outputs.strip()}\n")
    
    # font = ImageFont.truetype("Waree-Bold.ttf", 120)
    # draw = ImageDraw.Draw(images[0])
    # text = f"🚀 {self.model_name}: {outputs.strip()}"
    # # Calculate text size
    # text_size = draw.textlength(text, font=font)

    # # Calculate text position
    # x = int((images[0].width - text_size[0]) // 2)
    # y = int(images[0].height - 240 - text_size[1] // 2)

    # # Draw text on image
    # draw.text((x, y), text, font=font)

    # # Save the new image with the caption
    # images[0].save("/home/general/Documents/work/Trolls/blimp_v2/dreamerv3/outputs/img_example_with_caption.jpg")