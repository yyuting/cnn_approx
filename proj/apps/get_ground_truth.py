import tensorflow as tf
import os
import importlib
import sys; sys.path += ['../compiler']
from compiler import *
import skimage.io

img_lo_path = ["/bigtemp/yy2bb/images/Images_320_240"]
img_med_path = ["/bigtemp/yy2bb/images/Images_640_480"]
img_hi_path = ["/bigtemp/yy2bb/images/Images_1280_960"]
batch_size = 10

def get_ground_output(function, name):

    image_reader = tf.WholeFileReader()
    all_filenames = []
    for (paths, w, h) in [(img_lo_path, 320, 240), (img_med_path, 640, 480), (img_hi_path, 1280, 960)]:
    #for (paths, w, h) in [(['/localtmp/yuting/out_4_sample_random_features/zigzag_plane_normal_spheres/datas_each/train_label'], 640, 960)]:
        for path in paths:
            _, filename_prefix = os.path.split(path)
            write_filenames = []
            new_path = path + '_ground_' + name
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            nfiles = len([file for file in os.listdir(path) if file.endswith('.jpg')])
            img_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + '/*.jpg'), shuffle=False)
            img_name, img_file = image_reader.read(img_queue)
            img = tf.cast(tf.image.decode_jpeg(img_file), tf.float32) / 256.0
            img.set_shape((w, h, 3))
            name_batch, img_batch = tf.train.batch([img_name, img], batch_size)
            output = function(img_batch)
            sess = tf.Session()
            tf.local_variables_initializer().run(session=sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            batch_num = nfiles // batch_size
            for i in range(batch_num):
                (filenames, out_imgs) = sess.run([name_batch, output])
                for j in range(len(filenames)):
                    full_img_path = filenames[j].decode("utf-8")
                    _, filename = os.path.split(full_img_path) 
                    write_img = numpy.squeeze(out_imgs[j, :, :, :])
                    full_ground_path = os.path.join(new_path, filename)
                    skimage.io.imsave(full_ground_path, numpy.clip(write_img, 0.0, 1.0))
                    write_filenames.append(full_img_path + ' ' + full_ground_path)
            write_filenames.append('')
            all_filenames.append(filename_prefix+'_'+name+'.txt')
            open(filename_prefix+'_'+name+'.txt', 'w+').write('\n'.join(write_filenames))
    
    for all_filename in all_filenames:
        split_files(all_filename)

def split_files(filename):
    lines = open(filename).read().split('\n')[:-1]
    nfiles = len(lines)
    orig_name, ext = os.path.splitext(filename)
    ntrain = nfiles // 10 * 8
    ntest = nfiles // 10
    nvalidate = nfiles
    ans_train = '\n'.join(lines[:ntrain]) + '\n'
    ans_test = '\n'.join(lines[ntrain:ntrain+ntest]) + '\n'
    ans_validate = '\n'.join(lines[ntrain+ntest:]) + '\n'
    name_train = orig_name + 'train'
    name_test = orig_name + 'test'
    name_validate = orig_name + 'validate'
    open(name_train+ext, 'w+').write(ans_train)
    open(name_test+ext, 'w+').write(ans_test)
    open(name_validate+ext, 'w+').write(ans_validate)
            
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print('python get_ground_truth.py app_name gpu_name')
        print(' RGB only for now.')
        sys.exit(1)
        
    (app_name, gpu_name) = args[:2]
    
    input_module = importlib.import_module(app_name)
    objective = input_module.objective
    X = ImageRGB('x')
    c = CompilerParams(verbose=0, allow_g=False, check_save=False, sanity_check=False)
    
    (_, output_module) = get_module_prefix(objective(X), c)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    return get_ground_output(output_module.f, app_name)
    
def test():
    path = '/localtmp/yuting/out_4_sample_random_all_features/zigzag_plane_normal_spheres/datas_each/train_label'
    files1 = os.listdir('/localtmp/yuting/out_4_sample_random_all_features/zigzag_plane_normal_spheres/datas_each/train_label')
    files1 = [os.path.join('/localtmp/yuting/out_4_sample_random_all_features/zigzag_plane_normal_spheres/datas_each/train_label', file) for file in files1]
    files2 = os.listdir('/localtmp/yuting/out_4_sample_random_all_features/zigzag_plane_normal_spheres/datas_each/train_img')
    files2 = [os.path.join('/localtmp/yuting/out_4_sample_random_all_features/zigzag_plane_normal_spheres/datas_each/train_img', file) for file in files2]
    files1 = sorted(files1)
    files2 = sorted(files2)
    
    tensor1 = tf.convert_to_tensor(files1)
    tensor2 = tf.convert_to_tensor(files2)
    
    name1, name2 = tf.train.slice_input_producer([tensor1, tensor2], shuffle=True, num_epochs=1)
    arr = tf.reshape(tf.decode_raw(tf.read_file(name1), tf.float32), (640, 960, 1020))
    img = tf.to_float(tf.image.decode_png(tf.read_file(name2))) / 255.0
    
    sess = tf.Session()
    tf.local_variables_initializer().run(session=sess)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    for i in range(100):
        try:
            arr_name, img_name = sess.run([name1, name2])
            print(i, arr_name, img_name)
        except tf.errors.OutOfRangeError:
            print('caught successfully!')
    print('exit safely!')
    
if __name__ == '__main__':
    #main()
    test()