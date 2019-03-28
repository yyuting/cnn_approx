import numpy
import os
import skimage.io
import sys; sys.path += ['../compiler']
from compiler import *

total_tests = 7

test_start = 0
test_end = 7

test_case_size = 10

kernel_blur = numpy.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
kernel_sobel_x = numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
kernel_blur_large = 'kernel_blur_large.npy'

def objective0(X, Y):
    return X + X + X + X
test_cases0 = [{
'x': [1, 2, 3],
'y': [4, 5, 6]},
[4, 8, 12]]
    
def objective1(X, Y):
    return X * Y + X + Y
test_cases1 = [{
'x': [1, 2, 3],
'y': [4, 5, 6]},
[9, 17, 27]]
    
def objective2(X, Y):
    return X + 2.0 * Y + (X * Y)
test_cases2 = [{
'x': [1, 2, 3],
'y': [4, 5, 6]},
[13, 22, 33]]
    
def objective3(X, Y):
    return (X + Y) ** 2

def objective4(X, Y):
    filter = ConstArrayExpr(kernel_blur)
    return conv2d(X, filter)
    
def objective5(X, Y):
    sobel_x = ConstArrayExpr(kernel_sobel_x)
    sobel_y = array_transpose(sobel_x)
    gradient_x = conv2d(X, sobel_x)
    gradient_y = conv2d(X, sobel_y)
    mag = (gradient_x ** 2 + gradient_y ** 2) ** 0.5
    is_edge = mag >= 1
    return is_edge
    
def objective6(X, Y):
    """
    using ConstArrayExpr that read value from file
    """
    filter = ConstArrayExpr(numpy.load(kernel_blur_large))
    return conv2d(X, filter)
    
def test_objective(i):
    print("----------------------------------")
    print("testing objective", i)
    objectivename = 'objective%d'%i
    testcasename = 'test_cases%d'%i
    
    functor = eval(objectivename)
    try:
        test_cases = eval(testcasename)
    except:
        test_cases = None
        
    kw = {}
    
    c = CompilerParams(verbose=1)
    
    test_format = ['none']
    
    if i >= 3:
        test_format = ['gray', 'rgb']
        
    if i >= 4:
        c.allow_g = True
        
    for format in test_format:
        if format == 'none':
            X = ArgumentArray('x', shape=())
            Y = ArgumentArray('y', shape=())
        elif format == 'gray':
            X = ImageGray('x')
            Y = ImageGray('y')
        elif format == 'rgb':
            X = ImageRGB('x')
            Y = ImageRGB('y')
        else:
            raise ValueError('unknown format')
            
        if i == 3:
            input_shapes = numpy.random.randint(1, 100, [test_case_size, 2])
            x_input = []
            y_input = []
            out_input = []
            for k in range(test_case_size):
                random_shape = [1] + list(input_shapes[k, :]) + [1]
                if format == 'rgb':
                    random_shape[-1] = 3
                x_input.append(numpy.random.randn(*random_shape))
                y_input.append(numpy.random.randn(*random_shape))
                out_input.append((x_input[k] + y_input[k]) ** 2)
                
        if i == 4:
            if format == 'gray':
                (x_input, out_input) = images_from_file('blur', ['temple_gray'], multiples=2)
            elif format == 'rgb':
                (x_input, out_input) = images_from_file('blur', ['temple_rgb'], multiples=2)
            y_input = [None] * len(x_input)
            c.check_save = True
            kw['extra_checks'] = {check_unet: (APPROXNODE_PLACEHOLDER, x_input, x_input)}
        
        if i == 5:
            if format == 'rgb':
                continue
            (x_input, out_input) = images_from_file('edge', ['temple_gray'], multiples=2)
            y_input = [None] * len(x_input)
            c.check_save = True
            kw['extra_checks'] = {check_unet: (APPROXNODE_PLACEHOLDER, x_input, x_input)}
            
        if i == 6:
            if format == 'gray':
                (x_input, out_input) = images_from_file('blur_large', ['temple_gray'], multiples=2)
            elif format == 'rgb':
                (x_input, out_input) = images_from_file('blur_large', ['temple_rgb'], multiples=2)
            y_input = [None] * len(x_input)
            c.check_save = True
            kw['extra_checks'] = {check_unet: (APPROXNODE_PLACEHOLDER, x_input, x_input)}
        
        if test_cases is None:
            test_cases = [{
            'x': x_input,
            'y': y_input},
            out_input]
            
        f = functor(X, Y)
        success = check(f, c, test_cases=test_cases, **kw)
        
        if success is True:
            print("objective", i, format, "succeed")
        else:
            print("objective", i, format, "failed")
        print("----------------------------------")
        
        if c.check_save is True:
            rename_outputs(len(test_cases[1]), i, format)
        
        test_cases = None
        kw = {}
        
def check_unet(approxnode, train_inputs, test_inputs):
    ind_orig = approxnode.get_approx_ind('orig')
    train_outputs = approxnode.predict_approx(ind_orig, train_inputs, train_inputs)
    test_outputs = approxnode.predict_approx(ind_orig, test_inputs, test_inputs)
    
    ind_unet = approxnode.get_approx_ind('unet')
    approxnode.train_approx(ind_unet, train_inputs, train_outputs, iter=10)
    predict_loss = approxnode.predict_approx(ind_unet, test_inputs, test_outputs, 'loss')
    for loss in predict_loss:
        print(loss)
    return True
    
def rename_outputs(size, i, format):
    """
    rename output images written by output code using objective number and format
    """
    for k in range(size):
        src_name = 'out' + str(k) + '.png'
        dst_name = 'objective' + str(i) + '_' + format + '.png'
        os.rename(src_name, dst_name)
    
def images_from_file(operator, filenames, multiples=1):
    input_ans = []
    ground_ans = []
    for filename in filenames:
        input_filename = 'images/' + filename + '.png'
        input_array = read_float_image(input_filename)
        input_array = input_array[:input_array.shape[0]//multiples*multiples, :input_array.shape[1]//multiples*multiples]
        #input_ans.append(input_array)
        if operator is not None:
            ground_filename = 'images/' + filename + '_' + operator + '_ground.png'
            if os.path.exists(ground_filename):
                ground_array = read_float_image(ground_filename)
            else:
                operator_function = eval(operator + '_ground')
                ground_array = operator_function(input_array)
                skimage.io.imsave(ground_filename, numpy.clip(ground_array, 0.0, 1.0))
            ground_array = reshape_to_batch(ground_array)
            ground_ans.append(ground_array)
        input_array = reshape_to_batch(input_array)
        input_ans.append(input_array)
    return (input_ans, ground_ans)
    
def reshape_to_batch(array):
    """
    reshape array to fit batch tensor
    """
    if len(array.shape) == 2:
        array = numpy.expand_dims(array, axis=2)
    array = numpy.expand_dims(array, axis=0)
    return array
    
def read_float_image(filename):
    ans = skimage.io.imread(filename)
    return skimage.img_as_float(ans)
    
def main():
    for i in range(test_start, test_end):
        test_objective(i)
        
def img_conv(X, filter):
    """
    2D or 3D image convolving with 2D filter
    zero padding as in tensorflow
    """
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    x_size = filter.shape[0] // 2
    y_size = filter.shape[1] // 2
    w = X.shape[0]
    h = X.shape[1]
    out = numpy.zeros(X.shape)
    for r in range(w):
        for c in range(h):
            for x in range(filter.shape[0]):
                pixel_x = r + x - x_size
                if pixel_x < 0:
                    pixel_x = -pixel_x
                if pixel_x >= w:
                    pixel_x = w - pixel_x - 2
                for y in range(filter.shape[1]):
                    pixel_y = c + y - y_size
                    if pixel_y < 0:
                        pixel_y = -pixel_y
                    if pixel_y >= h:
                        pixel_y = h - pixel_y - 2
                    #if pixel_x >= 0 and pixel_x < w and pixel_y >= 0 and pixel_y < h:
                    out[r, c] += filter[x, y] * X[pixel_x, pixel_y]
    return out                
    
def blur_ground(X):
    """
    Ground truth code to compute objective4
    X is numpy array as input image
    rewritten form annotating_compiler/proj/apps/blur_one_stage/blur_one_stage.py
    """
    return img_conv(X, kernel_blur)
    
def blur_large_ground(X):
    kernel = numpy.load(kernel_blur_large)
    return img_conv(X, kernel)
    
def edge_ground(X):
    """
    Simple edge detection code for objective5
    """
    gradient_x = img_conv(X, kernel_sobel_x)
    gradient_y = img_conv(X, kernel_sobel_x.transpose())
    mag = (gradient_x ** 2.0 + gradient_y ** 2.0) ** 0.5
    is_edge = mag > 1.0
    return is_edge.astype('f')
    
def create_kernel_blur_large():
    kernel_size = 31
    kernel_half = kernel_size // 2
    sigma = 8
    import scipy.ndimage
    kernel_base = numpy.zeros([kernel_size, kernel_size])
    kernel_base[kernel_half, kernel_half] = 1.0
    kernel = scipy.ndimage.filters.gaussian_filter(kernel_base, sigma)
    numpy.save(kernel_blur_large, kernel)
    
if not os.path.exists(kernel_blur_large):
    create_kernel_blur_large()
    
if __name__ == '__main__':
    main()
    