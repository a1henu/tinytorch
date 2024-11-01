#include <iostream>

void im2col(
    const int* data_im,
    int* data_col,
    const int num_kernels,
    const int channels,
    const int height_col,
    const int width_col,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w) {
    for (int i = 0; i < num_kernels; ++i) {
        int* col_ptr = data_col;
        col_ptr += (i * kernel_h * kernel_w);

        int channel_index = i % channels;
        int off_channel = i / channels;
        int colw_index = off_channel % width_col;
        int colh_index = off_channel / width_col;

        int h_offset = colh_index * stride_h - pad_h;
        int w_offset = colw_index * stride_w - pad_w;

        for (int ix = 0; ix < kernel_h; ++ix) {
            for (int iy = 0; iy < kernel_w; ++iy) {
                int h_im = h_offset + ix * dilation_h;
                int w_im = w_offset + iy * dilation_w;

                *col_ptr =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im[((channel_index)*height + h_im) * width + w_im] : 0;

                col_ptr += 1;
            }
        }
    }
}

int main() {
    int data_im[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 2, 1, 6, 5, 4, 9, 8, 7};
    int data_col[32] = {0};

    int height = 3;
    int width = 3;
    int channels = 2;
    int kernel_h = 2;
    int kernel_w = 2;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    int num_kernels = channels * height_col * width_col;

    im2col(data_im, data_col, num_kernels, channels, height_col, width_col, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    for (int i = 0; i < 32; ++i) {
        std::cout << data_col[i] << " ";
    }
}