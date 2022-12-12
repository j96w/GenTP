import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from gentp.configs.base_config import Configs
from gentp.lib.model import SybTransModel
from gentp.lib.dataset import PreDataset

def pretrain(model, dataset, optimizer, train=True):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    torch.autograd.set_detect_anomaly(True)
    if train:
        model.train()
    else:
        model.eval()

    max_epoch = 200

    total_loss = 0.
    count = 0

    if train:
        for batch, data in enumerate(dataloader, 0):
            rgb, seg, syb, view_labl = data
            rgb, seg, syb, view_labl = Variable(rgb).cuda(), Variable(seg).cuda(), Variable(syb).cuda(), Variable(view_labl).cuda()

            output_syb, output_view, output_seg = model.forward_pre(rgb, seg, syb)

            label = syb[:, :, -1].view(-1).contiguous()
            view_labl = view_labl.view(-1).contiguous()
            seg_label = seg.view(-1).contiguous()
            output_seg = output_seg.view(-1, 2).contiguous()

            seg_negative_idx = (seg_label == 0).nonzero().view(-1)
            seg_positive_idx = (seg_label == 1).nonzero().view(-1)
            loss_seg_negative = 0
            loss_seg_positive = 0
            if len(seg_negative_idx) > 0:
                loss_seg_negative = criterion(output_seg[seg_negative_idx], seg_label[seg_negative_idx])
            if len(seg_positive_idx) > 0:
                loss_seg_positive = criterion(output_seg[seg_positive_idx], seg_label[seg_positive_idx])

            loss_syb = criterion(output_syb.view(-1, 2), label)
            loss_view = criterion(output_view.view(-1, 2), view_labl)

            loss = (loss_syb + loss_view + loss_seg_negative + loss_seg_positive) / 4.0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            print("Train step ", batch, loss_syb.item(), loss_view.item(), loss_seg_negative, loss_seg_positive)

            total_loss += loss.item()
            count += 1

            if count > max_epoch:
                break
    else:
        with torch.no_grad():
            for batch, data in enumerate(dataloader, 0):
                rgb, seg, syb, view_labl = data
                rgb, seg, syb, view_labl = Variable(rgb).cuda(), Variable(seg).cuda(), Variable(syb).cuda(), Variable(view_labl).cuda()

                output_syb, output_view, output_seg = model.forward_pre(rgb, seg, syb)

                label = syb[:, :, -1].view(-1).contiguous()
                view_labl = view_labl.view(-1).contiguous()
                seg_label = seg.view(-1).contiguous()
                output_seg = output_seg.view(-1, 2).contiguous()

                seg_negative_idx = (seg_label == 0).nonzero().view(-1)
                seg_positive_idx = (seg_label == 1).nonzero().view(-1)
                loss_seg_negative = 0
                loss_seg_positive = 0
                if len(seg_negative_idx) > 0:
                    loss_seg_negative = criterion(output_seg[seg_negative_idx], seg_label[seg_negative_idx])
                if len(seg_positive_idx) > 0:
                    loss_seg_positive = criterion(output_seg[seg_positive_idx], seg_label[seg_positive_idx])

                loss_syb = criterion(output_syb.view(-1, 2), label)
                loss_view = criterion(output_view.view(-1, 2), view_labl)

                loss = (loss_syb + loss_view + loss_seg_negative + loss_seg_positive) / 4.0

                print("Eval step ", batch, loss_syb.item(), loss_view.item(), loss_seg_negative, loss_seg_positive)

                total_loss += loss.item()
                count += 1

                if count > max_epoch:
                    break

    return total_loss / count

def main():
    configs = Configs()

    model = SybTransModel(configs.num_item, configs.num_act, configs.num_future_act, configs.emsize,
                          configs.nhead, configs.d_hid, configs.nlayers, configs.num_candidate, configs.image_size,
                          configs.num_syb, configs.object_num, configs.obj_id_list, configs.dropout).to(configs.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    data_pre_train = PreDataset(configs.buffer_size, configs.num_act, configs.num_future_act, 'train')
    data_pre_eval = PreDataset(configs.buffer_size, configs.num_act, configs.num_future_act, 'eval')

    best_val_loss = float('inf')
    for epoch in range(1, configs.epochs + 1):

        pretrain(model, data_pre_train, optimizer, True)
        val_loss = pretrain(model, data_pre_eval, optimizer, False)

        if (val_loss < best_val_loss) or (epoch % 10 == 0):
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), '{}/trained_models/pre_best_model_{}_{}.pth'.format(configs.base_path, epoch, val_loss))

if __name__ == "__main__":
    main()