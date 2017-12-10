from __future__ import division
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from msssim import *
from evaluation import *
import matplotlib.pyplot as plt
from discriminator import build_discriminator
import scipy.stats as st

EPS = 1e-12
task="reflection_removal_both_D/0004_good2/"
is_training=False
continue_training=True
hyper=True
# force to always use TITAN
#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
if is_training:
    os.environ['CUDA_VISIBLE_DEVICES']=str(0)
else:
    os.environ['CUDA_VISIBLE_DEVICES']=str(0)
#os.system('rm tmp')
k_diverse = 1
channel = 64

train_root=["/media/cecilia/DATA/reflection/train/"]
train_real_root=["/media/cecilia/DATA/reflection/train_real/"]
test_root=["/media/cecilia/DATA/reflection/test/"]
test_real_root=["/media/cecilia/DATA/reflection/test_real/"]

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias
    
def lrelu(x):
    return tf.maximum(x*0.2,x)

def relu(x):
    return tf.maximum(0.0,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

vgg_rawnet=scipy.io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
print("Loaded vgg19 pretrained imagenet")
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_rawnet['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

def build(input):
    if hyper:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
#    net=slim.conv2d(net,channel,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3*2,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return net

# def build_discriminator2(input, reuse=False):
#     net={}
#     with tf.variable_scope("D"):
#         if reuse:
#             tf.get_variable_scope().reuse_variables()
#         else:
#             assert tf.get_variable_scope().reuse == False
#         net["d_conv0"]=slim.conv2d(input,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=reuse,scope='d_conv0')
#         output = slim.batch_norm(net["d_conv0"],scope='d_bn0')
#         net["d_conv1"]=slim.conv2d(output,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=reuse,scope='d_conv1')
#         output = slim.batch_norm(net["d_conv1"],scope='d_bn1')
#         net["d_conv2"]=slim.conv2d(output,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=reuse,scope='d_conv2')
#         output = slim.batch_norm(net["d_conv2"],scope='d_bn2')
#         net["d_conv3"]=slim.conv2d(output,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=reuse,scope='d_conv3')
#         output = slim.batch_norm(net["d_conv3"],scope='d_bn3')
#         net["d_conv4"]=slim.conv2d(output,1,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),reuse=reuse,scope='d_conv4')
#         # flatten = slim.flatten(net)
#         # net=slim.fully_connected(flatten,1,activation_fn=None,reuse=reuse,scope='d_linear')
#     return tf.nn.sigmoid(net["d_conv4"]), net["d_conv4"], net

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel

g_mask=gkern(560,3)
g_mask=np.dstack((g_mask,g_mask,g_mask))

def syn_data(t,r,sigma):
    t=np.power(t,2.2)
    r=np.power(r,2.2)
    
    sz=int(2*np.ceil(2*sigma)+1)
    r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
    blend=r_blur+t
    
    att=1.08+np.random.random()/10.0
    
    for i in range(3):
        maski=blend[:,:,i]>1
        mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
        r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
    r_blur[r_blur>=1]=1
    r_blur[r_blur<=0]=0

    # alpha1 = 1-np.random.random()/5.0;
    h,w=r_blur.shape[0:2]
    neww=np.random.randint(0, 560-w-10)
    newh=np.random.randint(0, 560-h-10)
    alpha1=g_mask[newh:newh+h,neww:neww+w,:]
    alpha2 = 1-np.random.random()/5.0;
    r_blur_mask=np.multiply(r_blur,alpha1)
    blend=r_blur_mask+t*alpha2
    
    t=np.power(t,1/2.2)
    r_blur_mask=np.power(r_blur_mask,1/2.2)
    blend=np.power(blend,1/2.2)
    blend[blend>=1]=1
    blend[blend<=0]=0

    cv2.imwrite("/home/cecilia/Documents/tmp/%d_blend.jpg"%cnt,np.uint8(255*blend[:,:,0:3]))
    cv2.imwrite("/home/cecilia/Documents/tmp/%d_rblur.jpg"%cnt,np.uint8(255*r_blur[:,:,0:3]))
    cv2.imwrite("/home/cecilia/Documents/tmp/%d_t.jpg"%cnt,np.uint8(255*t[:,:,0:3]))
    cv2.imwrite("/home/cecilia/Documents/tmp/%d_alpha.jpg"%cnt,np.uint8(255*alpha1[:,:,0:3]))

    return t,r_blur_mask,blend

def prepare_data(train_path, test_path):
    input_names=[]
    output_names1=[]
    output_names2=[]
    # finetune_input_names=[]
    # finetune_output_names1=[]
    val_names=[]
    val_target1=[]
    val_target2=[]
    for dirname in train_path:
        train_t_gt = dirname + "transmission_layer/"
        train_r_gt = dirname + "reflection_layer/"
        train_b = dirname + "blended/"
        for root, _, fnames in sorted(os.walk(train_r_gt)):
            for fname in fnames:
                if is_image_file(fname):
                    path_input = os.path.join(train_b, fname)
                    path_output1 = os.path.join(train_t_gt, fname)
                    path_output2 = os.path.join(root, fname)
                    input_names.append(path_input)
                    output_names1.append(path_output1)
                    output_names2.append(path_output2)
    for dirname in test_path:
        test_t_gt = dirname + "transmission_layer/"
        test_r_gt = dirname + "reflection_layer/"
        test_b = dirname + "blended/"
        for root, _, fnames in sorted(os.walk(test_r_gt)):
            for fname in fnames:
                if is_image_file(fname):
                    path_input = os.path.join(test_b, fname)
                    path_output1 = os.path.join(test_t_gt, fname)
                    path_output2 = os.path.join(root, fname)
                    val_names.append(path_input)
                    val_target1.append(path_output1)
                    val_target2.append(path_output2)
    return input_names,output_names1,output_names2,val_names,val_target1,val_target2

###################################### Session
sess=tf.Session()
_,output_names1,output_names2,_,val_target1,val_target2=prepare_data(train_root,test_root)
input_real_names,output_real_names1,output_real_names2,val_real_names,val_real_target1,val_real_target2=prepare_data(train_real_root,test_real_root)
print(len(output_names1), input_real_names)

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_ms_ssim(img1, img2, cs_map=True):
    msssim_index= tf_ms_ssim(img1, img2, level=3)
    loss = (1.-msssim_index)/2.
    return loss

def compute_error(input,output):
    return tf.reduce_mean(tf.abs(output-input))

def compute_percep_loss(input, output, reuse=False):
    vgg_real=build_vgg19(output*255.0,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,reuse=True)
    p0=compute_error(vgg_real['input'],vgg_fake['input'])
    p1=compute_error(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_error(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_error(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_error(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_error(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return p0+p1+p2+p3+p4+p5

def compute_gradient_loss(img1,img2,level=1):
    gradx_loss=[]
    grady_loss=[]
    
    for l in range(level):
        gradx1, grady1=compute_gradient(img1)
        gradx2, grady2=compute_gradient(img2)
        alphax=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))
        alphay=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))
        
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*alphax)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*alphay)*2)-1
        # gradx1_s=tf.nn.sigmoid(tf.abs(gradx1))
        # grady1_s=tf.nn.sigmoid(tf.abs(grady1))
        # gradx2_s=tf.nn.sigmoid(tf.abs(gradx2*alphax))
        # grady2_s=tf.nn.sigmoid(tf.abs(grady2*alphay))

        gradx_loss.append(tf.reduce_mean(tf.multiply(tf.square(gradx1_s),tf.square(gradx2_s)),reduction_indices=[1,2,3])**0.25)
        grady_loss.append(tf.reduce_mean(tf.multiply(tf.square(grady1_s),tf.square(grady2_s)),reduction_indices=[1,2,3])**0.25)
        # gradx_loss.append(tf.reduce_mean(2*tf.sqrt(1e-6+tf.multiply(tf.abs(gradx1),tf.sqrt(1e-6+tf.abs(gradx2)))),reduction_indices=[1,2,3])*2**(level-1-l))
        # grady_loss.append(tf.reduce_mean(2*tf.sqrt(1e-6+tf.multiply(tf.abs(grady1),tf.sqrt(1e-6+tf.abs(grady2)))),reduction_indices=[1,2,3])*2**(level-1-l))
        img1=tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2=tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
    return gradx_loss,grady_loss

def compute_gradient(img):
    gradx=img[:,1:,:,:]-img[:,:-1,:,:]
    grady=img[:,:,1:,:]-img[:,:,:-1,:]
    return gradx,grady

def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,None,None,3])
    target=tf.placeholder(tf.float32,shape=[None,None,None,3])
    reflection=tf.placeholder(tf.float32,shape=[None,None,None,3])
    issyn = tf.placeholder(tf.bool,shape=[])

    network=build(input)
    transmission_layer, reflection_layer=tf.split(network, num_or_size_splits=2, axis=3)
    loss_percep_t=compute_percep_loss(transmission_layer, target)
    loss_percep_r=tf.where(issyn, compute_percep_loss(reflection_layer, reflection, reuse=True), 0.)
    loss_percep=tf.where(issyn, loss_percep_t+loss_percep_r, loss_percep_t)
    
    # Discriminator
    with tf.variable_scope("discriminator"):
        predict_real,pred_real_dict = build_discriminator(input,target)
    with tf.variable_scope("discriminator", reuse=True):
        predict_fake,pred_fake_dict = build_discriminator(input,transmission_layer)

    # D loss
    # d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(predict_real, tf.ones_like(predict_real)))
    # d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(predict_fake, tf.zeros_like(predict_fake)))
    # g_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(predict_fake, tf.ones_like(predict_fake)))
    d_loss = (tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))) * 0.5
    g_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
    
    # L1 loss
    loss_l1=compute_l1_loss(transmission_layer, target)
    loss_l1_r=tf.where(issyn,compute_l1_loss(reflection_layer, reflection),0)
    # SSIM Loss
    transmission_layer_r, transmission_layer_g, transmission_layer_b=tf.split(transmission_layer, num_or_size_splits=3, axis=3)
    output_r, output_g, output_b=tf.split(target, num_or_size_splits=3, axis=3)
    _ssim_r=compute_ms_ssim(output_r, transmission_layer_r)
    _ssim_g=compute_ms_ssim(output_g, transmission_layer_g)
    _ssim_b=compute_ms_ssim(output_b, transmission_layer_b)
    loss_ssim=(_ssim_r+_ssim_g+_ssim_b)/3.0
    # Gradient loss
    loss_gradx,loss_grady=compute_gradient_loss(transmission_layer,reflection_layer,level=3)
    loss_gradxy=tf.reduce_sum(sum(loss_gradx)/3.)+tf.reduce_sum(sum(loss_grady)/3.)
    loss_grad=tf.where(issyn,loss_gradxy/2.0,0)
    l1_ref=tf.where(issyn,tf.reduce_mean(tf.abs(reflection_layer)),0.0)

    loss=loss_l1+loss_l1_r+loss_percep*0.2+loss_ssim+loss_grad#-l1_ref*0.2

train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]
# var_list=[var for var in tf.trainable_variables() if 'discriminator' in var.name]
g_opt=tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss*100+g_loss, var_list=g_vars)
d_opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss,var_list=d_vars)
# check = tf.add_check_numerics_ops()

for var in tf.trainable_variables():
    print(var)

saver=tf.train.Saver(max_to_keep=20)
saver_restore=tf.train.Saver([var for var in tf.trainable_variables() if 'discriminator' not in var.name])
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(task)
print("contain checkpoint: ", ckpt)
if ckpt and continue_training:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)

maxepoch=20
k_sz=np.linspace(1,5,80)
rand_ratio=[0.5,1,2]
num_train=4000#len(output_names2)
g_mean=0
if is_training:
    all_l=np.zeros(num_train, dtype=float)
    all_l1=np.zeros(num_train, dtype=float)
    all_ssim=np.zeros(num_train, dtype=float)
    all_percep=np.zeros(num_train, dtype=float)
    all_grad=np.zeros(num_train, dtype=float)
    all_g=np.zeros(num_train, dtype=float)
    for epoch in range(1,maxepoch):
        # if epoch==1 or epoch==maxepoch+1:
        input_images=[None]*num_train
        output_images_t=[None]*num_train
        output_images_r=[None]*num_train

        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        cnt=0
        for id in np.random.permutation(num_train):
            st=time.time()
            if input_images[id] is None:
                magic=np.random.random()
                if magic < 1: # choose from synthetic dataset
                    is_syn=True
                    outputimg=cv2.imread(output_names1[id],-1)
                    neww=np.random.randint(256, 480)
                    newh=round((neww/outputimg.shape[1])*outputimg.shape[0])
                    output_image_t=cv2.resize(np.float32(outputimg),(neww,newh),cv2.INTER_CUBIC)/255.0
                    outputimg_r=cv2.resize(np.float32(cv2.imread(output_names2[id],-1)),(neww,newh),cv2.INTER_CUBIC)/255.0
                    file=os.path.splitext(os.path.basename(output_names1[id]))[0]
                    sigma=k_sz[np.random.randint(0, len(k_sz))]
                    if np.mean(output_image_t)*1/2 > np.mean(outputimg_r):
                        continue
                    output_image_t1,output_image_r,input_image=syn_data(output_image_t,outputimg_r,sigma)
                else: # choose from real dataste
                    is_syn=False
                    _id=id%len(input_real_names)
                    inputimg = cv2.imread(input_real_names[_id],-1)
                    file=os.path.splitext(os.path.basename(input_real_names[_id]))[0]
                    neww=np.random.randint(256, 480)
                    newh=round((neww/inputimg.shape[1])*inputimg.shape[0])
                    input_image=cv2.resize(np.float32(inputimg),(neww,newh),cv2.INTER_CUBIC)/255.0
                    output_image_t=cv2.resize(np.float32(cv2.imread(output_real_names1[_id],-1)),(neww,newh),cv2.INTER_CUBIC)/255.0
                    output_image_r=output_image_t # reflection gt not necessary
                    sigma=0.0
                input_images[id]=np.expand_dims(input_image,axis=0)
                output_images_t[id]=np.expand_dims(output_image_t,axis=0)
                output_images_r[id]=np.expand_dims(output_image_r,axis=0)
                
                if input_images[id].shape[1]*input_images[id].shape[2]>400000:#due to GPU memory limitation
                    continue
                if (input_images[id][:,:,0].sum() * input_images[id][:,:,1].sum() * input_images[id][:,:,2].sum()) < 1e-6:
                    print("Invalid file %s (degenerate channel)" % (file))
                    continue
                if (output_images_r[id][:,:,0].sum() * output_images_r[id][:,:,1].sum() * output_images_r[id][:,:,2].sum()) < 1e-6:
                    print("Invalid reflection file %s (degenerate channel)" % (file))
                    continue
                if input_images[id].max() < 0.2:
                    print("Invalid file %s (degenerate image)" % (file))
                    continue
                
                if cnt%2==0 and g_mean<1.5:
                    fetch_list=[d_opt]
                    # update D
                    _=sess.run(fetch_list,feed_dict={input:input_images[id],target:output_images_t[id]})
                fetch_list=[g_opt,transmission_layer,reflection_layer,
                    d_loss,g_loss,
                    loss,loss_l1,loss_ssim,loss_percep,loss_grad]
                # update G
                _,output_image_t,output_image_r,current_d,current_g,current,current_l1,current_ssim,current_percep,current_grad=\
                    sess.run(fetch_list,feed_dict={input:input_images[id],target:output_images_t[id],reflection:output_images_r[id],issyn:is_syn})
                all_l[id]=current
                all_l1[id]=current_l1*255
                all_ssim[id]=current_ssim
                all_percep[id]=current_percep
                all_grad[id]=current_grad*255
                all_g[id]=current_g
                g_mean=np.mean(all_g[np.where(all_g)])
                print("iter: %d %d || D: %.2f || G: %.2f %.2f || all: %.2f || loss: %.2f %.2f %.2f %.2f || mean: %.2f %.2f %.2f %.2f || time: %.2f"%
                    (epoch,cnt,current_d,current_g,g_mean,
                        np.mean(all_l[np.where(all_l)]),
                        current_l1*255,current_ssim,current_percep,current_grad*255,
                        np.mean(all_percep[np.where(all_percep)]),np.mean(all_l1[np.where(all_l1)]),np.mean(all_ssim[np.where(all_ssim)]),np.mean(all_grad[np.where(all_grad)]),
                        time.time()-st))
                cnt+=1
                # real_dict,fake_dict=sess.run([d_target_dict,d_output_dict],feed_dict={input:input_images[id],output:output_images_t[id],reflection:output_images_r[id],issyn:is_syn})
                # if cnt%2==0:
                    # cv2.imwrite("/home/ceciliazhang/Documents/tmp/%d_target_image_r.jpg"%cnt,np.uint8(255*output_images_r[id][0,:,:,0:3]))
                    # cv2.imwrite("/home/ceciliazhang/Documents/tmp/%d_input_image.jpg"%cnt,np.uint8(255*input_images[id][0,:,:,0:3]))
                    # cv2.imwrite("/home/cecilia/Documents/tmp/%d_d_conv2_r.jpg"%cnt,np.uint8(intensity_to_rgb(d_conv2_r)))
                    # cv2.imwrite("/home/cecilia/Documents/tmp/%d_d_conv2_f.jpg"%cnt,np.uint8(intensity_to_rgb(d_conv2_f)))
                input_images[id]=1.
                output_images_t[id]=1.
                output_images_r[id]=1.

        if epoch % 2 == 0:
            os.makedirs("%s/%04d"%(task,epoch))
            score=open("%s/%04d/score.txt"%(task,epoch),'w')
            score.write("Loss: %f\n"%np.mean(all_l1[np.where(all_l1)]))
            score.write("SSIM Loss: %f\n"%np.mean(all_ssim[np.where(all_ssim)]))
            score.write("Perceptual Loss: %f\n"%np.mean(all_percep[np.where(all_percep)]))

            saver.save(sess,"%s/model.ckpt"%task)
            saver.save(sess,"%s/%04d/model.ckpt"%(task,epoch))
            numtest=10
            all_val_l1=np.zeros(numtest, dtype=float)
            all_val_ssim=np.zeros(numtest, dtype=float)
            all_val_percep=np.zeros(numtest, dtype=float)
            for sigma in np.arange(1.5,5,0.5):
                for ind in range(numtest):
                    magic = np.random.random()
                    w_offset = np.random.randint(0, 128)
                    h_offset = np.random.randint(0, 128)
                    if magic < 0.5:
                        try:
                            target_image=np.float32(cv2.imread(val_target1[ind],-1))/255.0
                            targetimg_r=np.float32(cv2.imread(val_target2[ind],-1))/255.0
                        except:
                            continue
                        if target_image is None or targetimg_r is None:
                            continue
                        neww=np.random.randint(256, 480)
                        newh=round((neww/target_image.shape[1])*target_image.shape[0])
                        target_image=cv2.resize(np.float32(target_image),(neww,newh),cv2.INTER_CUBIC)/255.0
                        targetimg_r=cv2.resize(np.float32(targetimg_r),(neww,newh),cv2.INTER_CUBIC)/255.0
                        _,target_image_r,input_image=syn_data(target_image,targetimg_r,sigma)
                        st=time.time()
                        fetch_list=[transmission_layer,reflection_layer,loss_l1,loss_ssim,loss_percep]
                        output_image_t,output_image_r,all_val_l1[ind],all_val_ssim[ind],all_val_percep[ind]=\
                            sess.run(fetch_list,feed_dict={input:np.expand_dims(input_image,axis=0),target:np.expand_dims(target_image,axis=0),reflection:np.expand_dims(target_image_r,axis=0),issyn:True})
                        valid = "syn_"+os.path.splitext(os.path.basename(val_target1[ind]))[0]
                    else:
                        try:
                            inputimg=cv2.imread(val_real_names[ind],-1)
                            targetimg=cv2.imread(val_real_target1[ind],-1)
                        except:
                            continue
                        if inputimg is None or targetimg is None:
                            continue
                        neww=512
                        newh=round((neww/inputimg.shape[1])*inputimg.shape[0])
                        dim=(neww,newh)
                        input_image=np.float32(cv2.resize(inputimg,dim,cv2.INTER_CUBIC))/255.0
                        target_image=np.float32(cv2.resize(targetimg,dim,cv2.INTER_CUBIC))/255.0
                        target_image_r=target_image
                        st=time.time()
                        fetch_list=[transmission_layer,reflection_layer,loss_l1,loss_ssim,loss_percep]
                        output_image_t,output_image_r,all_val_l1[ind],all_val_ssim[ind],all_val_percep[ind]=\
                            sess.run(fetch_list,feed_dict={input:np.expand_dims(input_image,axis=0),target:np.expand_dims(target_image,axis=0),reflection:np.expand_dims(target_image,axis=0),issyn:False})
                        valid = "real_"+os.path.splitext(os.path.basename(val_real_names[ind]))[0]
                    if not os.path.isdir("%s/%04d/%s" % (task, epoch, valid)):
                        os.makedirs("%s/%04d/%s" % (task, epoch, valid))
                    print("test time for %s --> %.3f Loss: %f"%(valid,time.time()-st,all_val_l1[ind]*255.))
                    output_image_t=np.minimum(np.maximum(output_image_t,0.0),1.0)*255.0
                    output_image_r=np.minimum(np.maximum(output_image_r,0.0),1.0)*255.0
                    for i in range(k_diverse):
                        cv2.imwrite("%s/%04d/%s/t_%04d.jpg"%(task, epoch, valid, i),np.uint8(output_image_t[0,:,:,i*3:(i+1)*3]))
                        cv2.imwrite("%s/%04d/%s/r_%04d.jpg"%(task, epoch, valid, i),np.uint8(output_image_r[0,:,:,i*3:(i+1)*3]))
                score.write("sigma: %s\n"%sigma)
                score.write("Val Loss L1: %f\n"%np.mean(all_val_l1[np.where(all_val_l1)]*255.))
                score.write("Val Loss SSIM: %f\n"%np.mean(all_val_ssim[np.where(all_val_ssim)]))
                score.write("Val Loss Perceptual: %f\n"%np.mean(all_val_percep[np.where(all_val_percep)]))
            score.close()

test_gt=False
imgsz=256
subtask="test_video"
test_path = ["/media/cecilia/DATA/reflection/test_video/blended/"]
# test_path=["/home/cecilia/Documents/CEILNet/testdata_reflection_synthetic/"]
test_root_r_gt = "/media/cecilia/DATA/reflection/test_real/transmission_layer/"
test_root_gt = "/media/cecilia/DATA/reflection/test_real/reflection_layer/"

def prepare_data_test():
    input_names=[]
    output_names1=[]
    output_names2=[]
    # finetune_input_names=[]
    # finetune_output_names1=[]
    val_names=[]
    val_target1=[]
    val_target2=[]
    for dirname in test_path:
        for root, _, fnames in sorted(os.walk(dirname)):
            for fname in fnames:
                if is_image_file(fname):
                # if is_image_file(fname) and '-input' in fname:
                    path_input = os.path.join(root, fname)
                    path_output1 = os.path.join(test_root_gt, fname)
                    path_output2 = os.path.join(test_root_r_gt, fname)
                    val_names.append(path_input)
                    val_target1.append(path_output1)
                    val_target2.append(path_output2)
    return input_names,output_names1,output_names2,val_names,val_target1,val_target2

_,_,_,val_names,val_target1,val_target2=prepare_data_test()

if not os.path.isdir("%s/test_result2/%s"%(task,subtask)):
    os.makedirs("%s/test_result2/%s"%(task,subtask))
if test_gt:
    score=open("%s/test_result2/%s/"%(task,subtask)+"score.txt",'w')
out_mse=np.zeros(len(val_names), dtype=float)
out_ssim=np.zeros(len(val_names), dtype=float)
out_psnr=np.zeros(len(val_names), dtype=float)
for ind in range(len(val_names)):
    if not os.path.isfile(val_names[ind]):
        continue
    img_orig=cv2.imread(val_names[ind])
    if img_orig is None:
    	continue
    r=1 #imgsz/img_orig.shape[1]
    # dim = (imgsz,int(r*img_orig.shape[0]))
    # dim = (imgsz,imgsz)
    dim = (round(img_orig.shape[1]*r),round(img_orig.shape[0]*r))
    img=cv2.resize(img_orig,(dim),cv2.INTER_CUBIC)
    input_image=np.expand_dims(np.float32(img), axis=0)/255.0
    if input_image.shape[1]*input_image.shape[2]>400000: # due to GPU memory limitation
        continue
    st=time.time()
    output_image_t, output_image_r=sess.run([transmission_layer, reflection_layer],feed_dict={input:input_image})
    scale=[1,1,1]
    for i in range(3):
        scale[i]=np.sum(np.multiply(input_image[:,:,:,i],output_image_t[:,:,:,i]))/np.sum(np.multiply(output_image_t[:,:,:,i],output_image_t[:,:,:,i]))
        output_image_t[:,:,:,i]=output_image_t[:,:,:,i]*np.sqrt(scale[i])#np.sqrt(np.mean(scale))
    print("scale:",scale)
    output_image_t=np.minimum(np.maximum(output_image_t,0.0),1.0)*255.0
    output_image_r=np.minimum(np.maximum(output_image_r,0.0),1.0)*255.0
    testind = os.path.splitext(os.path.basename(val_names[ind]))[0]
    if not os.path.isdir("%s/test_result2/%s/%s" % (task,subtask,testind)):
        os.makedirs("%s/test_result2/%s/%s" % (task,subtask,testind))
    cv2.imwrite("%s/test_result2/%s/%s/input.jpg"%(task,subtask,testind),np.uint8(img))
    if test_gt:
        targetimg=cv2.imread(val_names[ind].replace("blended", "transmission_layer"))
        # targetimg=cv2.imread(val_names[ind].replace("-input", "-label1"),-1)
        if targetimg is None:
            continue
        im_ref_t = np.float32(cv2.resize(targetimg,dim,cv2.INTER_CUBIC))
        cv2.imwrite("%s/test_result2/%s/%s/target.jpg"%(task,subtask,testind),np.uint8(im_ref_t))
    # output_t=refine(output_image_t[0,:,:,i*3:(i+1)*3]/255.,input_image[0,:,:,i*3:(i+1)*3])
    cv2.imwrite("%s/test_result2/%s/%s/t_output.jpg"%(task,subtask,testind),np.uint8(output_image_t[0,:,:,0:3]))
    # cv2.imwrite("%s/test_result2/%d_t.jpg"%(task,ind),np.uint8(output_image_t[0,:,:,0:3]))
    # cv2.imwrite("%s/test_result2/%d_target.jpg"%(task,ind),np.uint8(im_ref_t))
    cv2.imwrite("%s/test_result2/%s/%s/r_output.jpg"%(task,subtask,testind),np.uint8(output_image_r[0,:,:,0:3]))
    if test_gt:
        print("%s\n"%val_names[ind])
        out_mse[ind] = compute_mse(im_ref_t/255., output_image_t[0,:]/255.)*65025.0
        out_ssim[ind] = compute_ssim(im_ref_t/255., np.uint8(output_image_t[0,:])/255.)
        out_psnr[ind] = compute_psnr(im_ref_t/255., np.uint8(output_image_t[0,:])/255.)
        score.write("%s: %f || %f || %f \n"%(val_names[ind],out_mse[ind],out_ssim[ind],out_psnr[ind]))
        print("%.3f, %s Loss: %f || %f || %f" % (time.time()-st, val_names[ind], out_mse[ind],out_ssim[ind],out_psnr[ind]))
        # print(np.mean(abs(output_image_t-im_ref_t)))
    else:
        print("%.3f, %s"%(time.time()-st, val_names[ind]))
if test_gt:
    score.write("Avg test loss: mse %f || ssim %f || psnr %f"%(np.mean(out_mse[np.where(out_mse)]), np.mean(out_ssim[np.where(out_ssim)]), np.mean(out_psnr[np.where(out_psnr)])))
    print("avg test loss mse %s || ssim %s || psnr %s"%(np.mean(out_mse[np.where(out_mse)]), np.mean(out_ssim[np.where(out_ssim)]), np.mean(out_psnr[np.where(out_psnr)])))

# plt.plot(list(range(len(val_names))),out_mse)
# plt.ylim((0,500))
# plt.ylabel('MSE')
# plt.xlabel('blur kernel')
# plt.show()
