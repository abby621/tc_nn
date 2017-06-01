from annoy import AnnoyIndex
import os
import shutil

def save_ims_and_index(ims,feats,save_directory):
    if len(ims) != len(feats):
        raise ValueError('Size of the image list does not match size of the feature list.')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    index_path = os.path.join(save_directory,'index.ann')
    if os.path.isfile(index_path):
        old_index_path = os.path.join(save_directory,'backup_index.ann')
        shutil.move(index_path,old_index_path)
    image_path = os.path.join(save_directory,'images.npy')
    if os.path.isfile(image_path):
        old_image_path = os.path.join(save_directory,'backup_images.npy')
        shutil.move(image_path,old_image_path)
    annoy_ind = AnnoyIndex(feats.shape[1])
    for ind,feat in zip(range(len(ims)),feats):
        print 'Adding image ', ims[ind]
        annoy_ind.add_item(ind,feat)
    annoy_ind.build(10) # build index w/ 10 trees
    annoy_ind.save(index_path)
    with open(image_path,'w+') as f:
        np.save(f,ims)
