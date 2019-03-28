import sys; sys.path += ['../compiler']
from compiler import *
import scipy.ndimage
import numpy
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

kernel_size = 5
kernel_half = kernel_size // 2
sigma = 1.5

def create_kernel():
    kernel_base = numpy.zeros([kernel_size, kernel_size])
    kernel_base[kernel_half, kernel_half] = 1.0
    kernel = scipy.ndimage.filters.gaussian_filter(kernel_base, sigma)
    return kernel
    
def objective(X):
    kernel = create_kernel()
    return conv2d(X, kernel)
    
def main():
    args = sys.argv[1:]
    if len(args) < 5:
        print('python test_blur_approx.py is_color iter saver_dir train_dir test_dir')
        sys.exit(1)
    (is_color, iter, saver_dir, train_dir, test_dir) = args[:5]
    is_color = bool(is_color)
    iter = int(iter)
    
    unet_kws = [
    {'nlayers': 2, 'feature_base': 8},
    {'nlayers': 3}]
    min_div = 4
    
    train_in = util.read_and_save_non_batch(train_dir, is_color, min_div)
    test_in = util.read_and_save_non_batch(test_dir, is_color, min_div)
    
    c = CompilerParams(verbose=1, allow_g=True, check_save=False, sanity_check=False)
    
    if is_color:
        X = ImageRGB('x')
    else:
        X = ImageGray('x')
    
    def test_unet_functor(unet_kw):
        def f(approxnode):
            ind_orig = approxnode.get_approx_ind('orig')
            train_out = approxnode.predict_approx(ind_orig, train_in, train_in)
            test_out = approxnode.predict_approx(ind_orig, test_in, test_in)
            
            ind_unet = approxnode.get_approx_ind('unet', **unet_kw)
            approxnode.train_approx(ind_unet, train_in, train_out, iter=iter, do_save=True, saver_dir=saver_dir)
            
            predict_output = approxnode.predict_approx(ind_unet, test_in, test_out)
            predict_loss = approxnode.predict_approx(ind_unet, test_in, test_out, 'loss')
            
            numpy.savez('test_blur_approx.npz', *predict_output)
            #for loss in predict_loss:
            #    print(loss)
            
            print("Training test finished, now test restoring the network")
            approxnode.clear_network(ind_unet)
            approxnode.train_approx(ind_unet, train_in, train_out, iter=0, read_from=saver_dir)
            predict_loss2 = approxnode.predict_approx(ind_unet, test_in, test_out, 'loss')
            
            for i in range(len(predict_loss)):
                assert util.is_eq(predict_loss[i], predict_loss2[i]), (predict_loss[i], predict_loss2[i])
            print("Restoring succeed")
            
            return True
        return f
        
    unet_kws = [
    {'nlayers': 2, 'feature_base': 8},
    {'nlayers': 3}]
            
    #extra_checks = {test_unet_functor({'nlayers': 2, 'feature_base': 8}): (APPROXNODE_PLACEHOLDER,)}
    extra_checks = {}
    for i in range(len(unet_kws)):
        extra_checks[test_unet_functor(unet_kws[i])] = (APPROXNODE_PLACEHOLDER,)
     
    check(objective(X), c, extra_checks=extra_checks)
    return
    
if __name__ == '__main__':
    main()