# 写`im2col`的正确姿势

我的代码中，`im2col`的签名为：
```cpp
template <typename Tp, typename Device>
struct im2col_op {
    /// @brief convert the image to the column matrix
    ///
    /// Inputs:
    /// @param device    : the type of device
    /// @param data_im   : the input image array pointer
    /// @param data_col  : the output column matrix array pointer
    /// @param channels  : the number of channels of the image
    /// @param height    : the height of the image
    /// @param width     : the width of the image
    /// @param kernel_h  : the height of the kernel
    /// @param kernel_w  : the width of the kernel
    /// @param pad_h     : the height of the padding
    /// @param pad_w     : the width of the padding
    /// @param stride_h  : the height of the stride
    /// @param stride_w  : the width of the stride
    void operator()(
        Device* device,
        const Tp* data_im,
        Tp* data_col,
        const int channels,
        const int height,
        const int width,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w
    );
};
```

这里用了一下trick来支持异构计算，暂且按下不表。这里主要讲一下`im2col`的实现。

## `im2col`

首先，极其重要的是：我的矩阵存储顺序是列优先！即：一个大小为`m`行`n`列的矩阵，第`(i, j)`元的索引为`i + j * m`

首先，我们要计算卷积后的feature map的大小。一个大小为`(h, w)`的图像，卷积核大小为`(kh, kw)`，padding为`(ph, pw)`，stride为`(sh, sw)`，那么卷积后的feature map的高度为`(h + 2 * ph - kh) / sh + 1`，宽度为`(w + 2 * pw - kw) / sw + 1`，分别记为`height_out`和`width_out`。

我首先计算了一下输出的`data_col`的维度，其每一行对应于一个卷积核的所有元素的展开，因此有`channels * kernel_h * kernel_w`个元素，也就是`channels * kernel_h * kernel_w`列，有`height_out * width_out`行。

对于每一行，我们先将第`i`个channel的对应位置的卷积核展开，之后再展开下一个channel的卷积核。

那么首先我们要对channel循环：
```cpp
for (int c = 0; c < channels; ++c) {
    // ...
}
```

之后，我们依次计算`data_col`的每一个元素，对于第`i`行，第`j`列的元素，它事实上是第`i`个卷积核内的第`j`个元素，我们可以计算出它在原图像中的位置。

首先，第`i`个卷积核在二维图像中事实上是第`(i / width_out, i % width_out)`个卷积核，因此它在原图像中的位置为`(i / width_out * stride_h - pad_h, i % width_out * stride_w - pad_w)`，分别记作`h_offset`和`w_offset`。

而一个卷积核中的第`j`个元素在卷积核中的位置为`(j / kernel_w, j % kernel_w)`，因此它在原图像中的位置为`(h_offset + j / kernel_w, w_offset + j % kernel_w)`

于是，我们可以得到
`data_col(i, j + c * kernel_h * kernel_w) = data_im(c, h_offset + j / kernel_w, w_offset + j % kernel_w)`

```cpp
template <typename Tp>
struct im2col_op<Tp, device::CPU> {
    void operator()(
        device::CPU* device,
        const Tp* data_im,
        Tp* data_col,
        const int channels,
        const int height,
        const int width,
        const int kernel_h,
        const int kernel_w,
        const int pad_h,
        const int pad_w,
        const int stride_h,
        const int stride_w
    ) {
        // calculate the size of the output img
        int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        // calculate the size of the col matrix(row-major) for each channel
        int width_col = kernel_h * kernel_w;
        int height_col = height_out * width_out;

        // for each channel
        for (int c = 0; c < channels; ++c) {
            const Tp* img = data_im + c * height * width;
            Tp* col = data_col + c * width_col * height_col;

            // for each point in the output col matrix
            for (int j = 0; j < width_col; ++j) {
                for (int i = 0; i < height_col; ++i) {
                    int h_offset = i / width_out * stride_h - pad_h;
                    int w_offset = i % width_out * stride_w - pad_w;

                    int kh_offset = j / kernel_w;
                    int kw_offset = j % kernel_w;

                    if (h_offset >= 0 && h_offset < height && w_offset >= 0 && w_offset < width) {
                        col[i + j * height_col] = img[h_offset + kh_offset + (w_offset + kw_offset) * height];
                    } else {
                        col[i + j * height_col] = 0;
                    }
                }
            }
        }
    }
};
```

这样，我们就完成了`im2col`的实现（希望我是对的）