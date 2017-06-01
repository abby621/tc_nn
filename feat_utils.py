def load_mean_from_binaryproto(filename):
    # mean_im = load_mean_from_binaryproto('/project/focus/datasets/tc_tripletloss/mean.binaryproto')
    import caffe
    from caffe.io import blobproto_to_array
    from caffe.proto import caffe_pb2
    import numpy as np
    blob = caffe_pb2.BlobProto()
    data = open(filename, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array(blobproto_to_array(blob))
    im_mean = arr.squeeze()
    return im_mean

def load_net(model,weights=None):
    # net = load_net('/project/focus/abby/tc_tripletloss/models/deploy/deploy.prototxt','/project/focus/abby/tc_tripletloss/models/alexnet_places365.caffemodel')
    import caffe
    if weights:
        net = caffe.Net(model,weights,caffe.TEST)
    else:
        net = caffe.Net(model,caffe.TEST)
    return net

def extract_features(im,mean_im,net,feat_layer):
    # feat = extract_features('/project/focus/datasets/hotel_chains/images/36313_284948_37_b.jpg',mean_im,net,'fc8')
    import caffe
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean_im)
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    orig_im = caffe.io.load_image(im)
    caffe_input= transformer.preprocess('data',orig_im)
    net.blobs['data'].data[...] = caffe_input
    out = net.forward()
    feat = net.blobs[feat_layer].data.copy().squeeze()
    return feat
