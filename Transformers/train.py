import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_target, max_len, device):
    sos_index = tokenizer_src.token_to_id('[SOS]')
    eos_index = tokenizer_src.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask) #(batch, seq_len, d_model)
    decoder_input = torch.tensor([[sos_index]], dtype=torch.int64).to(device) #(1, 1)
    while True:
        if decoder_input.size(1) >= max_len:
            break
        
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device) #(1, 1, seq_len, seq_len)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask) #(1, seq_len, d_model)

        prob = model.project(out[:, -1]) #(1, target_vocab_size)
        _, pred = torch.max(prob, dim = -1) #(1, 1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(pred.item()).to(device)], dim=1)

        if pred == eos_index:
            break
    
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_target, device, max_len, writer, global_state, print_msg, num_samples = 5):
    model.eval()

    count = 0
    source_text = []
    target_text = []
    pred_text = []

    console_width = 80

    with torch.no_grad():
        for batch in tqdm(validation_ds, desc=f'Validating'):
            count += 1
            encoder_input = batch['encoder_input'].to(device) #(Batch, seq_len)
            # decoder_input = batch['decoder_input'].to(device) #(Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(Batch, 1, 1, seq_len)
            # decoder_mask = batch['decoder_mask'].to(device) #(batch, 1, seq_len, seq_len)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_target, max_len, device) #(1, seq_len)

            source_text.append(batch['src_text'][0])
            target_text.append(batch['target_text'][0])
            pred_text.append(tokenizer_target.decode(model_output.detach().cpu().numpy()))

            print_msg('_'*console_width)
            print_msg(f'Source: {source_text[-1]}')
            print_msg(f'Target: {target_text[-1]}')
            print_msg(f'Pred: {pred_text[-1]}')
            print_msg('_'*console_width)

            if count >= num_samples:
                break
    
    if writer:
        pass



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):    
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    # train-val split = 90-10
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_target, config["lang_src"], config["lang_target"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_target, config["lang_src"], config["lang_target"], config["seq_len"])

    max_len_src, max_len_target = 0, 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print("Max length of src sequence ", max_len_src)
    print("Max length of target sequence ", max_len_target)

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target

def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_target = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading Model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]')).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch:02d}')
        
        for batch in batch_iterator:
            model.train()   
            encoder_input = batch['encoder_input'].to(device) #(Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device) #(Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) #(Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) #(batch, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) #(batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch, deq_len, d_model)
            proj_output = model.project(decoder_output) #(batch, seq_len, target_vocab_size)
            
            label = batch['label'].to(device) #(batch, seq_len)


            loss = loss_fn(proj_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)
        
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_target, device, config['seq_len'], writer, global_step, print_msg = print)

if __name__ == '__main__':

    config = get_config()
    train_model(config)
