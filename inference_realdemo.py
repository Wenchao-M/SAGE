import os
import random
import numpy as np
import torch
from tqdm import tqdm
import pickle
from utils import utils_transform
from utils.smplBody import BodyModel
from diffusion_stage.parser_util import get_args, merge_file
from dataloader.dataloader_refiner import load_data, RealDataset
from VQVAE.transformer_vqvae import TransformerVQVAE
from diffusion_stage.wrap_model import MotionDiffusion
from diffusion_stage.transformer_decoder import TransformerDecoder
from diffusion_stage.refinenet import Refinenet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################
lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
upper_body = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def overlapping_test_simplify(args, data, upper_model, model, vq_model_upper, vq_model_lower,
                              decoder_model, refinenet, num_per_batch=256):
    sparse_original, filename = data
    sparse_original = sparse_original.cuda().float()  # (seq, 54)
    num_frames = sparse_original.shape[0]

    gt_data_splits = []  # 重叠的序列
    block_seq = args.INPUT_MOTION_LENGTH  # 32
    seq_pad = sparse_original[:1].repeat(args.INPUT_MOTION_LENGTH - 1, 1)
    gt_data_pad = torch.cat((seq_pad, sparse_original), dim=0)  # (31+seq, 54)

    for i in range(num_frames):  # 需要预测的序列个数和序列长度相同
        gt_data_splits.append(gt_data_pad[i: i + block_seq])
    gt_data_splits = torch.stack(gt_data_splits)

    n_steps = gt_data_splits.shape[0] // num_per_batch
    if len(gt_data_splits) % num_per_batch > 0:
        n_steps += 1

    output_samples = []
    for step_index in range(n_steps):
        sparse_per_batch = gt_data_splits[step_index * num_per_batch: (step_index + 1) * num_per_batch].to(device)
        new_bs, new_seq = sparse_per_batch.shape[:2]
        # fullbody_res = torch.zeros((new_bs, new_seq, 22, 6)).to(device)
        with torch.no_grad():
            upper_latents = upper_model.diffusion_reverse(sparse_per_batch.reshape(new_bs, new_seq, 3, 18))
            lower_latents = model.diffusion_reverse(sparse_per_batch.reshape(new_bs, new_seq, 3, 18), upper_latents)
            recover_6d = decoder_model(upper_latents, lower_latents, sparse_per_batch)

        sample = recover_6d[:, -1].reshape(-1, 22 * 6)

        # sample = utils_transform.absSixd2rel_pavis_seq(sample)  # (seq, 132)
        # sample = sample.reshape(-1, 132)
        output_samples.append(sample.float())

    initial_res = torch.cat(output_samples, dim=0)
    pred_delta, hid = refinenet(initial_res[None], None)
    final_pred = pred_delta.squeeze() + initial_res
    head_motion = sparse_original[:, 12:15]
    return final_pred.cpu(), filename, head_motion


def test_process(args=None, log_path=None, cur_epoch=None):
    if args is None:
        cfg_args = get_args()
        cfg_args.cfg = 'config_decoder/refiner_realdemo.yaml'
        args = merge_file(cfg_args)
        name = cfg_args.cfg.split('/')[-1].split('.')[0]  # output directory
        args.SAVE_DIR = os.path.join("outputs", name)

    torch.backends.cudnn.benchmark = False
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    body_model = BodyModel(args.SUPPORT_DIR).to(device)

    print("Loading dataset...")
    filename_list, all_info = load_data(
        args.DATASET_PATH,
        "test",
        protocol=args.PROTOCOL
    )
    dataset = RealDataset(all_info, filename_list)

    # Load VQVAE model
    vqcfg = args.VQVAE
    vq_model_upper = TransformerVQVAE(in_dim=len(upper_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)
    vq_model_lower = TransformerVQVAE(in_dim=len(lower_body) * 6, n_layers=vqcfg.n_layers, hid_dim=vqcfg.hid_dim,
                                      heads=vqcfg.heads, dropout=vqcfg.dropout, n_codebook=vqcfg.n_codebook,
                                      n_e=vqcfg.n_e, e_dim=vqcfg.e_dim, beta=vqcfg.beta)

    # Load Diffusion model
    diff_model_upper = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_upper, use_upper=False).to(device)
    diff_model_lower = MotionDiffusion(cfg=args.DIFFUSION, input_length=args.INPUT_MOTION_LENGTH,
                                       num_layers=args.DIFFUSION.layers_lower, use_upper=True).to(device)
    decoder_model = TransformerDecoder(in_dim=132, seq_len=args.INPUT_MOTION_LENGTH, **args.DECODER).to(device)
    refineNet = Refinenet(n_layers=args.REFINER.n_layers, hidder_dim=args.REFINER.hidden_dim).to(device)

    # Upper VQVAE weight
    upper_vq_dir = args.UPPER_VQ_DIR
    vqvae_upper_file = os.path.join(upper_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_upper_file):
        checkpoint_upper = torch.load(vqvae_upper_file, map_location=lambda storage, loc: storage)
        vq_model_upper.load_state_dict(checkpoint_upper)
        print(f"=> Load upper vqvae {vqvae_upper_file}")
    else:
        print("No upper vqvae model!")
        return

    # Lower VQVAE weight
    lower_vq_dir = args.LOWER_VQ_DIR
    vqvae_lower_file = os.path.join(lower_vq_dir, 'best.pth.tar')
    if os.path.exists(vqvae_lower_file):
        checkpoint_lower = torch.load(vqvae_lower_file, map_location=lambda storage, loc: storage)
        vq_model_lower.load_state_dict(checkpoint_lower)
        print(f"=> Load upper vqvae {vqvae_lower_file}")
    else:
        print("No lower vqvae model!")
        return

    decoder_dir = args.FINETUNE_DIR
    decoder_file = os.path.join(decoder_dir, 'best.pth.tar')
    if os.path.exists(decoder_file):
        checkpoint_all = torch.load(decoder_file, map_location=lambda storage, loc: storage)
        diff_model_upper.load_state_dict(checkpoint_all['upper_state_dict'])
        diff_model_lower.load_state_dict(checkpoint_all['lower_state_dict'])
        decoder_model.load_state_dict(checkpoint_all['decoder_state_dict'])
        print("=> loading checkpoint '{}'".format(decoder_file))
    else:
        print("decoder file not exist!")
        return

    if hasattr(args, 'REFINER_DIR'):
        ourput_dir = args.REFINER_DIR
    else:
        ourput_dir = args.SAVE_DIR

    refine_file = os.path.join(ourput_dir, 'best.pth.tar')
    if os.path.exists(refine_file):
        print("=> loading refine model '{}'".format(refine_file))
        refine_checkpoint = torch.load(refine_file, map_location=lambda storage, loc: storage)
        refineNet.load_state_dict(refine_checkpoint)
    else:
        print(f"{refine_file} not exist!!!")
        return

    # torch.save(refine_checkpoint['state_dict'], refine_file)
    vq_model_upper.eval()
    vq_model_lower.eval()
    diff_model_upper.eval()
    diff_model_lower.eval()
    decoder_model.eval()
    refineNet.eval()

    test_loader = range(len(dataset))
    vis_save_dir = os.path.join("outputs", 'vis')
    os.makedirs(vis_save_dir, exist_ok=True)
    for sample_index in test_loader:
        output, filename, head_motion, = (
            overlapping_test_simplify(args, dataset[sample_index], diff_model_upper, diff_model_lower,
                                      vq_model_upper, vq_model_lower, decoder_model, refineNet, 1024))
        if isinstance(output, list):
            sample = torch.cat(output, dim=0)  # (N, 132)
        else:
            sample = output

        motion_pred = sample.squeeze().cuda()  # (seq_len, 132)

        model_rot_input = (  # (N, 66)
            utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach()).reshape(motion_pred.shape[0], -1).float()
        )
        t_head2world = head_motion.clone()
        body_pose_local = body_model(
            {
                "pose_body": model_rot_input[..., 3:66],
                "root_orient": model_rot_input[..., :3],
            }
        ).Jtr

        # Get the offset in global coordiante system between head and body_world.
        t_head2root = -body_pose_local[:, 15, :]  # root - head location
        t_root2world = t_head2root + t_head2world.cuda()

        # saving full body motion results to .pickle
        save_res = {
            "poses": model_rot_input.cpu().numpy(),
            "trans": t_root2world.cpu().numpy()
        }
        save_place = os.path.join(vis_save_dir, filename + '.pkl')
        file = open(save_place, 'wb')
        pickle.dump(save_res, file)
        file.close()
        print(f"Saving pickle to:{save_place}")


if __name__ == "__main__":
    test_process()
