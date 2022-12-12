import copy
import time
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from gentp.lib.model import SybTransModel
from gentp.lib.dataset import TransDataset
from gentp.configs.base_config import Configs

def train(data_buffer, model, optimizer, batch_size):
    data_buffer.switch_mode('train')
    dataloader = DataLoader(data_buffer, batch_size=batch_size, shuffle=True, num_workers=10)
    torch.autograd.set_detect_anomaly(True)
    criterion = nn.CrossEntropyLoss()

    model.train()  # turn on train mode
    max_epoch = 50
    count = 0

    for batch, data in enumerate(dataloader, 0):
        obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label = data
        obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label = Variable(obs).cuda(), \
                                                                                                                                                             Variable(base).cuda(), \
                                                                                                                                                             Variable(act).cuda(), \
                                                                                                                                                             Variable(succ).cuda(), \
                                                                                                                                                             Variable(score).cuda(), \
                                                                                                                                                             Variable(affordance_index).cuda(), \
                                                                                                                                                             Variable(affordance_select).cuda(), \
                                                                                                                                                             Variable(seg_id).cuda(), \
                                                                                                                                                             Variable(pre_syb).cuda(), \
                                                                                                                                                             Variable(post_syb).cuda(), \
                                                                                                                                                             Variable(segmentation).cuda(), \
                                                                                                                                                             Variable(segmentation_init).cuda(), \
                                                                                                                                                             Variable(future_syb).cuda(), \
                                                                                                                                                             Variable(future_act).cuda(), \
                                                                                                                                                             Variable(future_succ).cuda(), \
                                                                                                                                                             Variable(future_seg_id).cuda(), \
                                                                                                                                                             Variable(view_label).cuda()

        output_syb_pre, output_syb_post, output_view_select, act_pred, output_seg, output_all_syb, dis_feat = model.forward_traj(obs, base, act, affordance_index, affordance_select, pre_syb, seg_id, future_act, future_succ, future_seg_id, segmentation_init)

        loss_succ = criterion(act_pred.view(-1, 2), succ.view(-1)) * 0.3
        loss_view = criterion(output_view_select.view(-1, 2), affordance_index.view(-1)) * 0.3

        pre_syb_label = pre_syb[:, :, :].contiguous().view(-1)
        loss_syb_pre = criterion(output_all_syb.view(-1, 2), pre_syb_label)

        post_syb_label = post_syb[:, :, :].contiguous().view(-1)
        loss_syb_post = criterion(output_syb_post.view(-1, 2), post_syb_label)

        segmentation = segmentation.view(-1)
        output_seg = output_seg.view(-1, 2).contiguous()
        seg_negative_idx = (segmentation == 0).nonzero().view(-1)
        seg_positive_idx = (segmentation == 1).nonzero().view(-1)
        loss_seg_negative = 0
        loss_seg_positive = 0
        if len(seg_negative_idx) > 0:
            seg_negative_idx_subsample = seg_negative_idx
            loss_seg_negative = criterion(output_seg[seg_negative_idx_subsample], segmentation[seg_negative_idx_subsample])
        if len(seg_positive_idx) > 0:
            seg_positive_idx_subsample = seg_positive_idx
            loss_seg_positive = criterion(output_seg[seg_positive_idx_subsample], segmentation[seg_positive_idx_subsample])

        loss = loss_syb_pre + loss_syb_post + loss_view + loss_succ + loss_seg_negative + loss_seg_positive
        print("Train step ", batch, loss_syb_pre.item(), loss_syb_post.item(), loss_view.item(), loss_succ.item(), loss_seg_negative, loss_seg_positive, dis_feat.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        count += 1

        if count > max_epoch:
            break

def evaluate(data_buffer, model, eval_batch_size):
    data_buffer.switch_mode('test')
    dataloader = DataLoader(data_buffer, batch_size=eval_batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    count = 0.

    max_epoch = 80

    with torch.no_grad():
        for batch, data in enumerate(dataloader, 0):
            obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label = data
            obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label = Variable(obs).cuda(), \
                                                                                                                                                                            Variable(base).cuda(), \
                                                                                                                                                                            Variable(act).cuda(), \
                                                                                                                                                                            Variable(succ).cuda(), \
                                                                                                                                                                            Variable(score).cuda(), \
                                                                                                                                                                            Variable(affordance_index).cuda(), \
                                                                                                                                                                            Variable(affordance_select).cuda(), \
                                                                                                                                                                            Variable(seg_id).cuda(), \
                                                                                                                                                                            Variable(pre_syb).cuda(), \
                                                                                                                                                                            Variable(post_syb).cuda(), \
                                                                                                                                                                            Variable(segmentation).cuda(), \
                                                                                                                                                                            Variable(segmentation_init).cuda(), \
                                                                                                                                                                            Variable(future_syb).cuda(), \
                                                                                                                                                                            Variable(future_act).cuda(), \
                                                                                                                                                                            Variable(future_succ).cuda(), \
                                                                                                                                                                            Variable(future_seg_id).cuda(), \
                                                                                                                                                                            Variable(view_label).cuda()

            output_syb_pre, output_syb_post, output_view_select, act_pred, output_seg, output_all_syb, dis_feat = model.forward_traj(obs, base, act, affordance_index, affordance_select, pre_syb, seg_id, future_act, future_succ, future_seg_id, segmentation_init)

            loss_succ = criterion(act_pred.view(-1, 2), succ.view(-1)) * 0.3
            loss_view = criterion(output_view_select.view(-1, 2), affordance_index.view(-1)) * 0.3

            pre_syb_label = pre_syb[:, :, :].contiguous().view(-1)
            loss_syb_pre = criterion(output_all_syb.view(-1, 2), pre_syb_label)

            post_syb_label = post_syb[:, :, :].contiguous().view(-1)
            loss_syb_post = criterion(output_syb_post.view(-1, 2), post_syb_label)

            segmentation = segmentation.view(-1)
            output_seg = output_seg.view(-1, 2).contiguous()
            seg_negative_idx = (segmentation == 0).nonzero().view(-1)
            seg_positive_idx = (segmentation == 1).nonzero().view(-1)
            loss_seg_negative = 0
            loss_seg_positive = 0
            if len(seg_negative_idx) > 0:
                loss_seg_negative = criterion(output_seg[seg_negative_idx], segmentation[seg_negative_idx])
            if len(seg_positive_idx) > 0:
                loss_seg_positive = criterion(output_seg[seg_positive_idx], segmentation[seg_positive_idx])

            loss = loss_syb_pre + loss_syb_post + loss_view + loss_succ + loss_seg_negative + loss_seg_positive
            print("Test step ", batch, loss_syb_pre.item(), loss_syb_post.item(), loss_view.item(), loss_succ.item(), loss_seg_negative, loss_seg_positive, dis_feat.item())

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

    model.load_state_dict(torch.load(configs.pretrained_model))

    # Frozen pre-trained encoder and decoder networks
    for name, param in model.named_parameters():
        if name.split('.')[0] == 'encoder':
            param.requires_grad = False
        if name.split('.')[0] == 'trans_model' and name.split('.')[1] == 'segmentation_decoder':
            param.requires_grad = False
        if name.split('.')[0] == 'trans_model' and name.split('.')[1] == 'viewselector':
            param.requires_grad = False
        if name.split('.')[0] == 'symbol_decoder' and name.split('.')[1] == 'single_net' and (name.split('.')[2] == '3'): # frozen the OnTop symbolic decoders
            param.requires_grad = False
        if name.split('.')[0] == 'symbol_decoder' and name.split('.')[1] == 'double_net' and name.split('.')[2] == '0': # frozen the Cookable symbolic decoders
            param.requires_grad = False
        print(name, param.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr)

    data_buffer = TransDataset(configs.buffer_size, configs.num_act, configs.num_future_act)

    print("Data loaded!!!!!!!")

    best_val_loss = float('inf')
    for epoch in range(1, configs.epochs + 1):
        epoch_start_time = time.time()

        train(data_buffer, model, optimizer, configs.batch_size)
        val_loss = evaluate(data_buffer, model, configs.eval_batch_size)

        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ' f'valid loss {val_loss:5.2f}')
        print('-' * 89)

        if (val_loss < best_val_loss) or (epoch % 10 == 0):
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), '{}/trained_models/trans_best_model_{}_{}.pth'.format(configs.base_path, epoch, val_loss))

if __name__ == "__main__":
    main()