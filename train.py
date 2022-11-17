import time

from args import Arguments
from UNet import UNetVgg16
from datasets import get_dataloaders
from eval import eval_epoch
from utils import AverageMeter, ScoreMeter, Recorder, ModelSaver, LRScheduler, get_optimizer, get_loss_fn


def train(args):
    Arguments.save_args(args, args.args_path)
    train_loader, val_loader, _ = get_dataloaders(args)
    train_iter = iter(train_loader)
    model = UNetVgg16(n_classes=args.n_classes).to(args.device)
    optimizer = get_optimizer(args.optimizer, model)
    lr_scheduler = LRScheduler(args.lr_scheduler, optimizer)
    criterion = get_loss_fn(args.loss_type, args.ignore_index).to(args.device)
    model_saver = ModelSaver(args.model_path)
    recorder = Recorder(['train_miou', 'train_acc', 'train_loss',
                         'val_miou', 'val_acc', 'val_loss'])
    start = time.time()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(args.n_classes)
    print(f"number of batches in train dataloader: {len(train_loader)}")
    for iter_i in range(args.n_train_iters):
        # infinite loop over train_loader
        try:
            data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data = next(train_iter)
        inputs, labels, names = data
        inputs, labels = inputs.to(args.device), labels.long().to(args.device)
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = outputs.detach().cpu().numpy().argmax(axis=1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # measure
        loss_meter.update(loss.item(), inputs.size(0))
        score_meter.update(preds, labels.cpu().numpy())
        if (iter_i+1) % args.val_interval == 0:
            scores = score_meter.get_scores()
            train_loss, train_miou, train_ious, train_acc = loss_meter.avg, scores['mIoU'], scores['IoUs'], scores['accuracy']
            loss_meter.reset()
            score_meter.reset()

            val_loss, val_scores = eval_epoch(
                model=model,
                dataloader=val_loader,
                n_classes=args.n_classes,
                criterion=criterion,
                device=args.device,
            )
            val_miou, val_ious, val_acc = val_scores['mIoU'], val_scores['IoUs'], val_scores['accuracy']
            if (iter_i+1) % (args.val_interval * args.print_every_n_val) == 0:
                print(f"{args.experim_name} iter {iter_i+1} | "
                      f"train mIoU {train_miou:.3f} | accuracy {train_acc:.3f} | loss {train_loss:.3f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}", end=' | ')
                print(f"valid | mIoU {val_miou:.3f} | accuracy {val_acc:.3f} | loss {val_loss:.3f} | "
                      f"time {time.time() - start:.2f}", flush=True)
            recorder.update([train_miou, train_acc, train_loss, val_miou, val_acc, val_loss])
            recorder.save(args.record_path)
            if args.metric.startswith("IoU"):
                metric = val_ious[int(args.metric.split('_')[1])]
            else: metric = val_miou
            model_saver.save_models(metric, iter_i+1, model,
                                    ious={'train': train_ious, 'val': val_ious})

    print(f"best model at iter {model_saver.best_iter} with miou {model_saver.best_score:.5f}")


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args(verbose=True)
    train(args)
