import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from maml import STMAML
from tqdm import tqdm

def train_epoch(train_dataloader):
    train_losses = []
    for step, (data, A_wave) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        A_wave = A_wave.to(device=args.device)
        A_wave = A_wave.float()
        data = data.to(device=args.device)
        out, meta_graph = model(data, A_wave)
        loss_predict = loss_criterion(out, data.y)
        loss_reconsturct = loss_criterion(meta_graph, A_wave)
        loss = loss_predict + loss_reconsturct
        loss.backward()
        optimizer.step()
        # print("loss_predict: {}, loss_reconsturct: {}".format(loss_predict.detach().cpu().numpy(), loss_reconsturct.detach().cpu().numpy()))
        train_losses.append(loss.detach().cpu().numpy())
    return sum(train_losses)/len(train_losses)

def test_epoch(test_dataloader):
    with torch.no_grad():
        model.eval()
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=args.device)
            data = data.to(device=args.device)
            out, _ = model(data, A_wave)
            if step == 0:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label


parser = argparse.ArgumentParser(description='MAML-based')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='metr-la', type=str)
parser.add_argument('--source_epochs', default=200, type=int)
parser.add_argument('--source_lr', default=1e-2, type=float)
parser.add_argument('--target_epochs', default=120, type=int)
parser.add_argument('--target_lr', default=1e-2, type=float)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--meta_dim', default=16, type=int)
parser.add_argument('--target_days', default=3, type=int)
parser.add_argument('--model', default='GRU', type=str)
parser.add_argument('--loss_lambda', default=1.5, type=float)
parser.add_argument('--memo', default='revise', type=str)
args = parser.parse_args()

print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)

if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.load(f)

    torch.manual_seed(7)

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    
    model_args['meta_dim'] = args.meta_dim
    model_args['loss_lambda'] = args.loss_lambda
    
    source_dataset = traffic_dataset(data_args, task_args, "source", test_data=args.test_dataset)

    model = STMAML(data_args, task_args, model_args, model=args.model).to(device=args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.source_lr)
    loss_criterion = nn.MSELoss()

    source_training_losses, target_training_losses = [], []
    best_result = ''
    min_MAE = 10000000

    for epoch in tqdm(range(args.source_epochs)):
        # Meta-Train
        start_time = time.time()
        spt_task_data, spt_task_A, qry_task_data, qry_task_A = source_dataset.get_maml_task_batch(task_args['task_num'])
        loss = model.meta_train_revise(spt_task_data, spt_task_A, qry_task_data, qry_task_A)

        # loss = model.meta_train(spt_task_data, spt_task_A, qry_task_data, qry_task_A)
        end_time = time.time()
        if epoch % 10 == 0:
            print("[Source Train] epoch #{}/{}: loss is {}, training time is {}".format(epoch+1, args.source_epochs, loss, end_time-start_time))

    print("Source dataset meta-train finish.")

    target_dataset = traffic_dataset(data_args, task_args, "target", test_data=args.test_dataset, target_days=args.target_days)
    target_dataloader = DataLoader(target_dataset, batch_size=task_args['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    test_dataset = traffic_dataset(data_args, task_args, "test", test_data=args.test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=8, pin_memory=True)

    model.finetuning(target_dataloader, test_dataloader, args.target_epochs)
    print(args.memo)
    