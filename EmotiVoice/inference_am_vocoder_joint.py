# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import os, sys, warnings, torch, glob, argparse
import numpy as np
from models.hifigan.get_vocoder import MAX_WAV_VALUE
import soundfile as sf
from yacs import config as CONFIG
from tqdm import tqdm
import traceback
def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.no_grad():
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = os.path.join(config.output_directory, args.logdir)
    ckpt_path = os.path.join(root_path,  "ckpt")
    files = os.listdir(ckpt_path)
    
    for file in files:
        if args.checkpoint:
            if file != args.checkpoint:
                continue

        checkpoint_path = os.path.join(ckpt_path, file)

        with open(config.model_config_path, 'r') as fin:
            conf = CONFIG.load_cfg(fin)
        
     
        conf.n_vocab = config.n_symbols
        conf.n_speaker = config.speaker_n_labels

        style_encoder = StyleEncoder(config)
        model_CKPT = torch.load(config.style_encoder_ckpt, map_location="cpu")
        model_ckpt = {}
        for key, value in model_CKPT['model'].items():
            new_key = key[7:]
            model_ckpt[new_key] = value
        style_encoder.load_state_dict(model_ckpt, strict=False)



        generator = JETSGenerator(conf).to(device)

        model_CKPT = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(model_CKPT['generator'])
        generator.eval()

        with open(config.token_list_path, 'r') as f:
            token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

        with open(config.speaker2id_path, encoding='utf-8') as f:
            speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}


        tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        
        text_path = args.test_file

   
        if os.path.exists(root_path + "/test_audio/audio/" +f"{file}/"):
            r = glob.glob(root_path + "/test_audio/audio/" +f"{file}/*")
            for j in r:
                os.remove(j)
        texts = []
        prompts = []
        speakers = []
        contents = []
        with open(text_path, "r") as f:
            for line in f:
                print(f"Processing line: {line.strip()}")
                line = line.strip().split("|")
                speakers.append(line[0])
                prompts.append(line[1])
                texts.append(line[2].split())
                contents.append(line[3])

        print(f"Processing {contents}")
        for i, (speaker, prompt, text, content) in enumerate(tqdm(zip(speakers, prompts, texts, contents), total=len(speakers))): # Add total to tqdm

            print(f"\n--- Processing Line {i+1} ---")
            print(f"  Speaker: {speaker}, Prompt: {prompt}")
            # print(f"  Phonemes: {' '.join(text)}") # Uncomment for verbose phoneme logging
            # print(f"  Content: {content}") # Uncomment for verbose content logging

            try:
                # --- Get Embeddings (Numpy Arrays) ---
                style_embedding_np = get_style_embedding(prompt, tokenizer, style_encoder)
                content_embedding_np = get_style_embedding(content, tokenizer, style_encoder)

                # Check if embeddings are valid
                if style_embedding_np is None or content_embedding_np is None:
                    print(f"Error: Failed to get style or content embedding for line {i+1}. Skipping.")
                    continue

                # --- Validate Speaker ---
                if speaker not in speaker2id:
                    print(f"Warning: Speaker ID '{speaker}' not found for line {i+1}. Skipping.")
                    continue
                speaker_id_int = speaker2id[speaker]

                # --- Convert Phonemes to IDs ---
                try:
                    text_int = [token2id[ph] for ph in text]
                except KeyError as e:
                    print(f"Error: Phoneme '{e}' not found in token list for line {i+1}. Skipping.")
                    print(f"  Problematic phoneme list: {text}")
                    continue # Skip this line if a phoneme is invalid

                # --- Convert Data to Tensors ---
                # Use the numpy arrays from above
                sequence = torch.from_numpy(np.array(text_int)).to(device).long().unsqueeze(0)
                sequence_len = torch.from_numpy(np.array([len(text_int)])).to(device)
                style_embedding_tensor = torch.from_numpy(style_embedding_np).to(device).unsqueeze(0)
                content_embedding_tensor = torch.from_numpy(content_embedding_np).to(device).unsqueeze(0)
                speaker_tensor = torch.from_numpy(np.array([speaker_id_int])).to(device)

                print(f"  Inputs prepared for model (line {i+1}).")

                # --- Generate Audio ---
                with torch.no_grad():
                    # Generation and Saving Try Block
                    try:
                        print(f"  Calling generator model (line {i+1})...")
                        infer_output = generator(
                                inputs_ling=sequence,
                                inputs_style_embedding=style_embedding_tensor, # Use tensor
                                input_lengths=sequence_len,
                                inputs_content_embedding=content_embedding_tensor, # Use tensor
                                inputs_speaker=speaker_tensor, # Use tensor
                                alpha=1.0
                            )
                        print(f"  Generator call successful (line {i+1}).")

                        audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
                        audio = audio.cpu().numpy().astype('int16')
                        print(f"  Audio data processed (line {i+1}).")

                        # Define output directory/path
                        output_dir = os.path.join(root_path, "test_audio", "audio", file) # 'file' is checkpoint name
                        output_path = os.path.join(output_dir, f"{i+1}.wav")

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)

                        # Save the audio file
                        print(f"  Saving audio to: {output_path}")
                        sf.write(file=output_path, data=audio, samplerate=config.sampling_rate)
                        print(f"  Successfully saved: {output_path}")

                    except Exception as e_gen:
                        print(f"!! Error during generation or saving for line {i+1} !!")
                        print("------------------- Traceback Start -------------------")
                        traceback.print_exc() # Print the full traceback
                        print("-------------------- Traceback End --------------------")
                        continue # Skip to next line on generation error

            except Exception as e_outer:
                # Catch any unexpected errors outside the inner generation block
                print(f"!! Outer loop error processing line {i+1} !!")
                print("------------------- Traceback Start -------------------")
                traceback.print_exc() # Print the full traceback
                print("-------------------- Traceback End --------------------")
                continue # Attempt to continue with the next line

        print("--- Loop Finished ---") # Add a marker for loop completion





if __name__ == '__main__':
    print("run!")
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--logdir', type=str, required=True)
    p.add_argument("-c", "--config_folder", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=False, default='', help='inference specific checkpoint, e.g --checkpoint checkpoint_230000')
    p.add_argument('-t', '--test_file', type=str, required=True, help='the absolute path of test file that is going to inference')

    args = p.parse_args() 
    ##################################################
    sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "/" + args.config_folder)

    from config import Config
    config = Config()
    ##################################################
    main(args, config)



