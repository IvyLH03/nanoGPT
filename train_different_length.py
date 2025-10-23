
import subprocess
import os
import shutil
import numpy as np

base_dataset = 'lovecraft'
base_data_dir = os.path.join(base_dataset)


def eval_length(data_path, out_dir):
    """
    Sample from a trained model
    """
    import os
    import pickle
    from contextlib import nullcontext
    import numpy as np
    import torch
    import tiktoken
    from model import GPTConfig, GPT

    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 10 # number of samples to draw
    max_new_tokens = 1000 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        print(f"Loading model from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode training text for evaluation
    # load from data/shakespeare_char/val.bin and split it into reasonable batches
    with open(os.path.join(data_path, 'val.bin'), 'rb') as f:
        val_data = np.fromfile(f, dtype=np.uint16)
    val_data = torch.from_numpy(val_data.astype(np.int64)).to(device)
    def get_batch(split, batch_size, block_size):
        data = val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y




    # run evaluation
    with torch.no_grad():
        with ctx:
            # perplexity evaluation
            total_perplexity = 0.0
            for _ in range(num_samples):
                x, y = get_batch('val', 1, model.config.block_size)
                logits, loss = model(x, y)
                perplexity = torch.exp(loss)
                total_perplexity += perplexity.item()
                print(f"Perplexity: {perplexity.item():.2f}")
            avg_perplexity = total_perplexity / num_samples
            print(f"Average Perplexity over {num_samples} samples: {avg_perplexity:.2f}")

            # distinct-n evaluation
            generated_texts = []
            context = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]
            generated = model.generate(context, max_new_tokens, temperature=temperature, top_k=top_k)[0]
            generated_text = decode(generated.tolist())
            generated_texts.append(generated_text)
            
            def distinct_n(texts, n):
                ngrams = set()
                total_ngrams = 0
                for text in texts:
                    tokens = text.split()
                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i:i+n])
                        ngrams.add(ngram)
                        total_ngrams += 1
                return len(ngrams) / total_ngrams if total_ngrams > 0 else 0.0
            
            for n in range(1, 5):
                dn = distinct_n(generated_texts, n)
                print(f"Distinct-{n}: {dn:.4f}")

            # load the input text, build a set of words, and calculate the percentage of words in the sample output that actually appears in the input text
            with open(os.path.join('data', 'lovecraft', 'input.txt'), 'r') as f:
                input_text = f.read()
            input_words = set(input_text.split())
            output_words = set(generated_text.split())
            overlap = output_words.intersection(input_words)
            print(f"Word overlap: {len(overlap)}/{len(output_words)} = {len(overlap) / len(output_words):.4f}")

    return avg_perplexity, len(overlap) / len(output_words), distinct_n(generated_texts,1), distinct_n(generated_texts,2), distinct_n(generated_texts,3), distinct_n(generated_texts,4)



# Other fixed arguments for train.py
base_args = [
    'train.py',
    'config/lovecraft.py',
    '--device=mps',
    '--compile=False',
    '--eval_iters=20',
    '--log_interval=1',
    '--block_size=64',
    '--batch_size=12',
    '--n_layer=40',
    '--n_head=8',
    '--n_embd=128',
    '--max_iters=2000',
    '--lr_decay_iters=2000',
    '--dropout=0.0'
]

avg_perplexity_dict = {}
overlap_dict = {}
distinct_1_dict = {}
distinct_2_dict = {}
distinct_3_dict = {}
distinct_4_dict = {}

train_size = 10000
while True:
    data_path = os.path.join(base_data_dir, f'{train_size}')
    output_path = os.path.join('out-lovecraft', f'{train_size}')
    os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(os.path.join('data',data_path)):
        break

    args = ['python'] + base_args + [f'--dataset={data_path}', f'--out_dir={output_path}']
    print(f"Running training with args: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    print(f"Finished train_size={train_size}, exit code: {result.returncode}")
    # if result.stdout:
    #     print("Output:")
    #     print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

    avg_perplexity, overlap, distinct_1, distinct_2, distinct_3, distinct_4 = eval_length(os.path.join('data', data_path), output_path)
    avg_perplexity_dict[train_size] = avg_perplexity
    overlap_dict[train_size] = overlap
    distinct_1_dict[train_size] = distinct_1
    distinct_2_dict[train_size] = distinct_2
    distinct_3_dict[train_size] = distinct_3
    distinct_4_dict[train_size] = distinct_4

    train_size += 10000

# print results
print("Final Results:")
print("Training Size\tAvg Perplexity\tWord Overlap\tDistinct-1\tDistinct-2\tDistinct-3\tDistinct-4")
for size in sorted(avg_perplexity_dict.keys()):
    print(f"{size}\t{avg_perplexity_dict[size]:.2f}\t{overlap_dict[size]:.4f}\t{distinct_1_dict[size]:.4f}\t{distinct_2_dict[size]:.4f}\t{distinct_3_dict[size]:.4f}\t{distinct_4_dict[size]:.4f}")

# graph the results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(list(avg_perplexity_dict.keys()), list(avg_perplexity_dict.values()), marker='o')
plt.title('Average Perplexity')
plt.xlabel('Training Size')
plt.ylabel('Perplexity')

plt.subplot(2, 3, 2)
plt.plot(list(overlap_dict.keys()), list(overlap_dict.values()), marker='o')
plt.title('Word Overlap')
plt.xlabel('Training Size')
plt.ylabel('Overlap')

plt.subplot(2, 3, 3)
plt.plot(list(distinct_1_dict.keys()), list(distinct_1_dict.values()), marker='o')
plt.title('Distinct-1')
plt.xlabel('Training Size')
plt.ylabel('Distinct-1')

plt.subplot(2, 3, 4)
plt.plot(list(distinct_2_dict.keys()), list(distinct_2_dict.values()), marker='o')
plt.title('Distinct-2')
plt.xlabel('Training Size')
plt.ylabel('Distinct-2')

plt.subplot(2, 3, 5)
plt.plot(list(distinct_3_dict.keys()), list(distinct_3_dict.values()), marker='o')
plt.title('Distinct-3')
plt.xlabel('Training Size')
plt.ylabel('Distinct-3')

plt.subplot(2, 3, 6)
plt.plot(list(distinct_4_dict.keys()), list(distinct_4_dict.values()), marker='o')
plt.title('Distinct-4')
plt.xlabel('Training Size')
plt.ylabel('Distinct-4')

plt.tight_layout()
plt.show()