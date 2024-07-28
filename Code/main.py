from Training import *
import argparse

def main(opts):

    train(opts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')

    # data loader related
    parser.add_argument('--dataroot', type=str, required=True, help='path of data')
    parser.add_argument('--num_domains', type=int, default=3)
    parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--input_dim', type=int, default=1, help='# of input channels')
    parser.add_argument('--nThreads', type=int, default=8, help='# of threads for data loader')

    # ouptput related
    parser.add_argument('--name', type=str, default='trial', help='folder name to save outputs')
    parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    parser.add_argument('--img_save_freq', type=int, default=5, help='freq (epoch) of saving images')
    parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

    # training related
    parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
    parser.add_argument('--dis_norm', type=str, default='None', help='normalization layer in discriminator [None, Instance]')
    parser.add_argument('--dis_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    parser.add_argument('--n_ep', type=int, default=1200, help='number of epochs') # 400 * d_iter
    parser.add_argument('--n_ep_decay', type=int, default=600, help='epoch start decay learning rate, set -1 if no decay')
    parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    parser.add_argument('--d_iter', type=int, default=3, help='# of iterations for updating content discriminator')
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_cls', type=float, default=3.0)
    parser.add_argument('--lambda_cls_G', type=float, default=10.0)
    parser.add_argument('--isDcontent', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    opts = parser.parse_args()

    main(opts)