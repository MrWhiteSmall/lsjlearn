

def load_cifar10_from_tar(tar_path):
    import tarfile
    import pickle
    import numpy as np
    """直接从tar.gz文件读取，不解压到磁盘"""
    
    data_batches = {}
    '''
    [
    <TarInfo 'cifar-10-batches-py' at 0x7f1d46a1d000>, 
    <TarInfo 'cifar-10-batches-py/readme.html' at 0x7f1d46a1d9c0>, 
labels>>
    <TarInfo 'cifar-10-batches-py/batches.meta' at 0x7f1d46a1dc00>, 
test>>
    <TarInfo 'cifar-10-batches-py/test_batch' at 0x7f1d46a1da80>, 
train>>
    <TarInfo 'cifar-10-batches-py/data_batch_1' at 0x7f1d46a1de40>
    <TarInfo 'cifar-10-batches-py/data_batch_2' at 0x7f1d46a1dcc0>, 
    <TarInfo 'cifar-10-batches-py/data_batch_3' at 0x7f1d46a1db40>, 
    <TarInfo 'cifar-10-batches-py/data_batch_4' at 0x7f1d46a1d900>, 
    <TarInfo 'cifar-10-batches-py/data_batch_5' at 0x7f1d46a1d840>, 
    ]
    '''
    with tarfile.open(tar_path, 'r:gz') as tar:
        # 读取所有文件
        for member in tar.getmembers():
            if member.name.endswith('.html'):
                continue
                
            file = tar.extractfile(member)
            if file:
                data = pickle.load(file, encoding='bytes')
                data_batches[member.name] = data
    
    # 处理训练数据
    train_data = []
    train_labels = []
    
    for i in range(1, 6):
        batch_key = f'cifar-10-batches-py/data_batch_{i}'
        batch_data = data_batches[batch_key][b'data']        # numpy (10000, 3072)
        batch_labels = data_batches[batch_key][b'labels']    # list (10000,)
        
        train_data.append(batch_data)
        train_labels.extend(batch_labels)
    
    train_data = np.vstack(train_data)  # (50000, 3072)
    train_labels = np.array(train_labels)
    
    # 处理测试数据
    test_key = 'cifar-10-batches-py/test_batch'
    test_data = data_batches[test_key][b'data']      # (10000, 3072)
    test_labels = np.array(data_batches[test_key][b'labels'])
    
    # 获取标签名称
    meta_key = 'cifar-10-batches-py/batches.meta'
    '''
    array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck'], dtype='<U10')
    '''
    label_names = [name.decode('utf-8') for name in data_batches[meta_key][b'label_names']]
    
    return (train_data, train_labels), (test_data, test_labels), label_names

# 使用示例
# (train_data, train_labels), (test_data, test_labels), label_names = load_cifar10_from_tar('datasets/cifar-10-python.tar.gz')

from torch.utils.data import Dataset, DataLoader

class CIFAR10FromTar(Dataset):
    def __init__(self, tar_path, train=True, transform=None):
        self.transform = transform
        
        # 加载数据
        if train:
            (self.data, self.labels), _, _ = load_cifar10_from_tar(tar_path)
        else:
            _, (self.data, self.labels), _ = load_cifar10_from_tar(tar_path)
        
        # 重塑为图像格式 (50000, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (50000, 32, 32, 3)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]  # (32, 32, 3)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def test():
    # 创建数据加载器
    dataset = CIFAR10FromTar('datasets/cifar-10-python.tar.gz', train=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        image, label = batch
        break