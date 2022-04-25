import tensorflow as tf
import cv2
import numpy as np
import os
import albumentations as A
from hparams import hparams

"""### Augmentation method"""

transform = A.Compose([
    # A.ColorJitter (p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

def load_np(predict_file):
    predict_dict = np.load(predict_file, allow_pickle=True).item()
    h, w = list(predict_dict.values())[0].shape
    tensor = np.zeros((21,h,w),np.float32)
    for key in predict_dict.keys():
        tensor[key+1] = predict_dict[key]
    tensor[0,:,:] = 0.26
    predict = np.argmax(tensor, axis=0).astype(np.uint8)
    return predict

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, mode='train'):
        self.mode = mode
        self.bs = hparams['batch_size']
        self.im_shape = hparams['image_shape']
        paths = hparams['train_path'] if mode == 'train' else hparams['val_path']
        self.data = [{'img': os.path.join(paths['image'], name[:-3]+'jpg'),
                 'mask': os.path.join(paths['mask'], name)}
                for name in os.listdir(paths['mask'])]   
        self.categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

            
    def __len__(self):
        return len(self.data) // self.bs
    
    def __getitem__(self, idx):
        start = idx*self.bs
        batch = self.data[start:start + self.bs]
        h, w = self.im_shape[:2]
        images, labels = [], []
        for d in batch:
#             print(d['img'], d['mask'])
            src_img = cv2.imread(d['img'])
            img = cv2.resize(src_img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if self.mode == 'train':
            #     mask = load_np(d['mask'])
            # else:
            mask = cv2.imread(d['mask'], cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (w, h))
            # convert mask to onehot version
            mask = tf.one_hot(mask, 21).numpy()
            images.append(img)
            labels.append(mask)
            # if self.mode == 'train':
            #     #Apply data augmentation
            #     transformed = transform(image=img, mask=mask)
            #     transformed_image = transformed['image']
            #     transformed_mask = transformed['mask']
            #     images.append(transformed_image)
            #     labels.append(transformed_mask)
            
        images, labels = np.array(images)/255.0, np.array(labels, dtype=np.float32)
#         print(images.shape, labels.shape)
        return images, labels
    
    def on_epoch_end(self):
        np.random.shuffle(self.data)
        
if __name__ == '__main__':
#     train_gen = Dataset('train')
    val_gen = Dataset('val')
    for i, b in enumerate(val_gen):
        print(b[0].shape, b[1])
    