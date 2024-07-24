import torch

def saveepochcheckpont(args, epoch, net, optimizer=None):
    # savepath_best = args.output_dir  + '/best_' + args.dataset_name + '_c2f_tcn.wt'
    savepath = args.output_dir + f'/{epoch}ep__' + args.dataset_name + '_c2f_tcn.wt'
    #     print('Saving checkpoint...')
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     if not os.path.isdir('checkpoint/' + netname):
    #         os.mkdir('checkpoint/' + netname)
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    # torch.save(state, savepath_best)
    torch.save(state, savepath)

def saveepochcheckpont_best(args, epoch, net, optimizer=None):
    savepath_best = args.output_dir  + '/best_' + args.dataset_name + '_c2f_tcn.wt'
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, savepath_best)

def saveepochcheckpont_semi(args, epoch, net, prefix):
    savepath = args.output_dir + "/" + prefix + f'{epoch}ep__' + args.dataset_name + '_c2f_tcn.wt'
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, savepath)


def saveepochcheckpont_semi_best(args, epoch, net, prefix):
    savepath_best =args.output_dir + "/" + prefix + 'best_' + args.dataset_name + '_c2f_tcn.wt'
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, savepath_best)

def saveepochcheckpont_semi_best_model_d(args, epoch, net, prefix):
    savepath_best =args.output_dir + "/" + prefix + 'best_' + args.dataset_name + '_c2f_tcn_model_d.wt'
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, savepath_best)

def load_best_model(args):
    checkpoint =  torch.load(args.output_dir+'/best_' + args.dataset_name + '_c2f_tcn.wt')
    # checkpoint =  torch.load(args.output_dir + '/50ep__' + args.dataset_name + '_c2f_tcn.wt')
    # checkpoint =  torch.load(args.output_dir + '/35ep__' + args.dataset_name + '_c2f_tcn.wt')
    epoch = checkpoint['epoch']
    print("test epoch----", epoch)


    return checkpoint['net']

def load_unsupervised_model(model_path):
    checkpoint =  torch.load(model_path)
    epoch = checkpoint['epoch']
    print("test epoch----", epoch)
    return checkpoint['net']

def load_best_model_semi(args, prefix=""):
    checkpoint = torch.load(args.output_dir + "/" + prefix + 'best_' + args.dataset_name+ '_c2f_tcn.wt')
    # checkpoint = torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn.wt')
    epoch = checkpoint['epoch']
    print("load best model-->epoch----", epoch)
    return checkpoint['net']

def load_best_model_semi_model_d(args, prefix=""):
    checkpoint = torch.load(args.output_dir + "/" + prefix + 'best_' + args.dataset_name+ '_c2f_tcn_model_d.wt')
    # checkpoint = torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn_model_d.wt')
    epoch = checkpoint['epoch']
    print("load best model_d-->epoch----", epoch)
    return checkpoint['net']

def load_best_model_semi_after_niter(args, prefix='unsuper'):
    # checkpoint = torch.load(args.output_dir + "/" + prefix + '10ep__' + args.dataset_name + '_c2f_tcn.wt')
    # checkpoint = torch.load(args.output_dir + "/" + prefix + 'best_' + args.dataset_name + '_c2f_tcn.wt')
    checkpoint = torch.load(args.output_dir + "/" + 'best_' + args.dataset_name + '_c2f_tcn.wt')
    # checkpoint = torch.load(args.output_dir + "/" + '50ep__' + args.dataset_name + '_c2f_tcn.wt')

    epoch = checkpoint['epoch']
    print("test epoch----", epoch)
    return checkpoint['net']

def load_best_unsupervised_model(args, prefix='unsuper'):
    print('==>load best unsupervised checkpoint..')
    # checkpoint = torch.load(args.output_dir + "/" + prefix + '10ep__' + args.dataset_name + '_c2f_tcn.wt')
    checkpoint = torch.load(args.output_dir + "/" + prefix + 'best_' + args.dataset_name + '_c2f_tcn.wt')
    epoch = checkpoint['epoch']
    print("test epoch----", epoch)
    return checkpoint['net']


def resume_checkpoint(args, net, model_d, optimizer=None):
    if  args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # checkpoint = torch.load(args.output_dir  + '/378ep__' + args.dataset_name + '_c2f_tcn.wt')
        checkpoint = torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn.wt')
        checkpoint2 = torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2f_tcn_model_d.wt')
        net.load_state_dict(checkpoint['net'])
        model_d.load_state_dict(checkpoint2['net'])
        start_epoch = checkpoint['epoch'] + 1
        print("start_epoch----", start_epoch)

        return start_epoch, net, model_d, optimizer
    else:
        return 0, net, model_d, optimizer

def resume_checkpoint2(args, net, optimizer=None):
    if  args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.output_dir  + '/best_' + args.dataset_name + '_c2f_tcn.wt')
        # checkpoint = torch.load(args.output_dir + '/200ep__' + args.dataset_name + '_c2f_tcn.wt')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print("start_epoch----", start_epoch)

        return start_epoch, net, optimizer
    else:
        return 0, net, optimizer