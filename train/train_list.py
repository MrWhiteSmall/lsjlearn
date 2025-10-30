import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train.alexnet import train_alexnet
from train.resnet import train_resnet

# INSERT_YOUR_CODE
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train models on CIFAR-10')
    parser.add_argument('-c', '--cuda', type=str, default='2', help='CUDA device id')
    
    subparsers = parser.add_subparsers(title='models', dest='model')

    # AlexNet
    parser_alexnet = subparsers.add_parser('alexnet', help='Train AlexNet on CIFAR-10')
    parser_alexnet.add_argument('--tar_path', type=str, default='datasets/cifar-10-python.tar.gz')
    parser_alexnet.add_argument('--num_classes', type=int, default=10)
    parser_alexnet.add_argument('--batch_size', type=int, default=128)
    parser_alexnet.add_argument('--num_epochs', type=int, default=20)
    parser_alexnet.add_argument('--lr', type=float, default=0.01)
    parser_alexnet.add_argument('--weight_decay', type=float, default=5e-4)
    parser_alexnet.add_argument('--device', type=str, default=None)
    parser_alexnet.add_argument('--save_path', type=str, default='alexnet_cifar10.pth')
    parser_alexnet.set_defaults(func=lambda args: train_alexnet(
        tar_path=args.tar_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device if args.device is None else args.device,
        save_path=args.save_path
    ))

    # ResNet
    parser_resnet = subparsers.add_parser('resnet', help='Train ResNet on CIFAR-10')
    parser_resnet.add_argument('--tar_path', type=str, default='datasets/cifar-10-python.tar.gz')
    parser_resnet.add_argument('--num_classes', type=int, default=10)
    parser_resnet.add_argument('--batch_size', type=int, default=128)
    parser_resnet.add_argument('--num_epochs', type=int, default=40)
    parser_resnet.add_argument('--lr', type=float, default=0.01)
    parser_resnet.add_argument('--weight_decay', type=float, default=5e-4)
    parser_resnet.add_argument('--device', type=str, default=None)
    parser_resnet.add_argument('--save_path', type=str, default='resnet_cifar10.pth')
    parser_resnet.set_defaults(func=lambda args: train_resnet(
        tar_path=args.tar_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device if args.device is None else args.device,
        save_path=args.save_path
    ))

    args = parser.parse_args()

    # 额外处理 --cuda/-c 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
