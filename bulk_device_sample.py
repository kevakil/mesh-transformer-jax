import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm

from google.cloud import storage
import os

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args

def float_to_string(float_num):
    return str(round(float_num, 1))

def findMiddle(input_list):
    print(input_list)
    return input_list[len(input_list)//2]

if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]
    total_steps = params["total_steps"]
    ckpt_every = params["ckpt_every"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')



    """
    # a diverse set of prompts to test the effectiveness of the models
    [Spongebob pets Gary]\nGary:  - to test if gary can meow
    Squidward: Another day, another migraine - to test how much it's overfitting
    Shrek: What are you doing in my swamp?! - to test out-of-sample characters
    Patrick: I'm gunna lick your juicy asshole Spongebob!\nSpongebob: [on the verge of orgasm] Aww fuck yeah, Pat! Lick daddy's bussy like you mean it!\nSquidward: [shocked] - to test literotica lol
    [The next morning, Patrick and Squidward are at the gym. They are shocked to see Larry injecting steroids] - guided scene generation
    [Patrick won't stop farting]\nSquidward: [unamused] - potential to be funny
    [The night before the Fry Cook Games, Spongebob and Sandy are working on their deadlifts. Larry and Plankton, now best friends, enter the gym. Mr. Krab's advice is still stuck in Spongebob's head. Patrick and Old Man Jenkins watch using their binoculars. Patrick's binoculars are backwards] - to test complex character interactions

    # extra overfitting tests
    Squidward: [says to himself] Open 24 hours a day. What a stupid idea. Who wants a Krabby Patty at 3 in the morning?\n[cuts to Patrick's bedroom.\nPatrick's alarm clock goes off.]\nPatrick: [turns off the alarm] Oh, boy! 3 A.M.! [whips out a Krabby Patty from under his blanket and starts to eat it; cuts back to The Krusty Krab] - strong overfitting test

    Nosferatu: [as a bat]- ? overfitting test, out-of-sample factual knowledge test

    <|endoftext|>\n - unprompted generation
    """
    
    prompts = [
        'Gary:',
        'Squidward: Another day, another migraine',
        'Shrek: What are you doing in my swamp?!',
        "Patrick: I\'m gunna lick your juicy asshole Spongebob!\nSpongebob: [on the verge of orgasm] Aww fuck yeah, Pat! Lick daddy\'s bussy like you mean it!\nSquidward: [shocked]",
        "[The next morning, Patrick and Squidward are at the gym. They are shocked to see Larry injecting steroids]",
        "[Patrick won't stop farting]\nSquidward: [unamused]",
        "[The night before the Fry Cook Games, Spongebob and Sandy are working on their deadlifts. Larry and Plankton, now best friends, enter the gym. Mr. Krab's advice is still stuck in Spongebob's head. Patrick and Old Man Jenkins watch using their binoculars. Patrick's binoculars are backwards]",
        "Squidward: [says to himself] Open 24 hours a day. What a stupid idea. Who wants a Krabby Patty at 3 in the morning?\n[cuts to Patrick's bedroom.\nPatrick's alarm clock goes off.]\nPatrick: [turns off the alarm] Oh, boy! 3 A.M.! [whips out a Krabby Patty from under his blanket and starts to eat it; cuts back to The Krusty Krab]",
        "Nosferatu: [as a bat]",
        "<|endoftext|>\n"

    ]

    encoded_prompts = []
    for prompt in prompts:
        encoded_prompts.append(tokenizer.encode(prompt))


    # sweep through different levels of dogma (ckpt_steps) and temperature
    # should i sweep through top_p as well?
    # i guess it cant hurt
    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    # shit, i forgot, cloud objects are techinically not stored in directories, so this theoretically wouldnt work. still there doesnt seem to be a builtin for this on gcs and i have no idea why...
    # annoying implementation because it depends on the config file being right
    ckpt_steps = [total_steps] + list(range(total_steps+1, 0, -ckpt_every))[1:]

    # i change this line to manually perform a binary search for the right number of epochs to use (balance between correctness and not overfitting too much)
    ckpt_steps = ckpt_steps[len(ckpt_steps) // 2:]
    ckpt_steps = [findMiddle(ckpt_steps)]
    print('chkpt steps', ckpt_steps)

    # sweep through checkpoints
    for ckpt_step in ckpt_steps:
        print(f"using checkpoint {ckpt_step}")

        total_batch = per_replica_batch * jax.device_count() // cores_per_replica
        with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
            network = CausalTransformer(params)

            start = time.time()
            network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
            print(f"network loaded in {time.time() - start:.06}s")

            local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
            del network.state["opt_state"]
            network.state = network.move_xmap(network.state, np.zeros(local_shards))


            # sweep through temps, top_p
            # this might produce too many prompt-completions, do the math!
            for top_p_amount in [0.5]: #[0.25, 0.5, 0.75]:# np.arange(0.2, 1.1, 0.2):
                for temp_amount in [0.5, 1, 1.5]:#np.arange(0.2, 2.1, 0.2):
                    outfile_path = f"samples/ckpt-{ckpt_step}/temp-{float_to_string(temp_amount)}/top_p-{float_to_string(top_p_amount)}.txt"
                    text = ''

                    # create the directories if they dont exist
                    os.makedirs(os.path.dirname(outfile_path), exist_ok=True)

                    with open(outfile_path, 'w') as out:
                        for tokens in encoded_prompts:
                            start = time.time()

                            provided_ctx = len(tokens)
                            pad_amount = seq - provided_ctx

                            padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
                            batched_tokens = np.array([padded_tokens] * total_batch)
                            length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

                            print("~~~~~~~~~~~~~~~GENERATING~~~~~~~~~~~~~~~")
                            output = network.generate(batched_tokens, length, pad_amount // 2, {"top_p": np.ones(total_batch) * top_p_amount,
                                                                            "temp": np.ones(total_batch) * temp_amount})

                            orig_input = tokenizer.decode(tokens)


                            for idx, o in enumerate(output[1][0][:, :, 0]):
                                outtext = repr(tokenizer.decode(o))
                                print(f"sample {idx}: {outtext}")
                                print(orig_input)
                                print(" => ")
                                print(outtext)
                                text += orig_input + outtext +"\n\n===\n\n"

                            print(f"completion done in {time.time() - start:06}s")
                        out.write(text)
                        print('writing to', outfile_path)
