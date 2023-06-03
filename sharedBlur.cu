/*By eliminating redundant calculations and simplifying the memory access pattern, the blur program's performance can be enhanced. To reduce global memory accesses and make input data available to numerous threads, we specifically used tiling. The input image is divided into tiles so that each thread can focus on a different area of the image while reusing information from surrounding threads. As a result, the number of global memory accesses is decreased, and the memory access pattern is enhanced. We can see this in the run times of both programs, the program using tiling and shared memory was significantly faster than the naive method.
 */
#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>

#define BLUR_SIZE 16 // size of surrounding image is 2X this
#define TILE_SIZE 16

#include "bitmap_image.hpp"

using namespace std;

__global__ void blurKernel(uchar3 *in, uchar3 *out, int width, int height)
{
    __shared__ uchar3 tile[TILE_SIZE + 2 * BLUR_SIZE][TILE_SIZE + 2 * BLUR_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int bx = threadIdx.x + BLUR_SIZE;
    int by = threadIdx.y + BLUR_SIZE;

    // copy the input tile into shared memory
    if (col < width && row < height)
    {
        tile[by][bx] = in[row * width + col];

        // pad the borders with neighboring pixels
        if (threadIdx.x < BLUR_SIZE)
        {
            tile[by][bx - BLUR_SIZE] = in[row * width + max(col - BLUR_SIZE, 0)];
            tile[by][bx + TILE_SIZE] = in[row * width + min(col + TILE_SIZE, width - 1)];
        }

        if (threadIdx.y < BLUR_SIZE)
        {
            tile[by - BLUR_SIZE][bx] = in[max(row - BLUR_SIZE, 0) * width + col];
            tile[by + TILE_SIZE][bx] = in[min(row + TILE_SIZE, height - 1) * width + col];
        }

        if (threadIdx.x < BLUR_SIZE && threadIdx.y < BLUR_SIZE)
        {
            tile[by - BLUR_SIZE][bx - BLUR_SIZE] = in[max(row - BLUR_SIZE, 0) * width + max(col - BLUR_SIZE, 0)];
            tile[by - BLUR_SIZE][bx + TILE_SIZE] = in[max(row - BLUR_SIZE, 0) * width + min(col + TILE_SIZE, width - 1)];
            tile[by + TILE_SIZE][bx - BLUR_SIZE] = in[min(row + TILE_SIZE, height - 1) * width + max(col - BLUR_SIZE, 0)];
            tile[by + TILE_SIZE][bx + TILE_SIZE] = in[min(row + TILE_SIZE, height - 1) * width + min(col + TILE_SIZE, width - 1)];
        }
    }

    __syncthreads();

    if (col < width && row < height)
    {
        int3 Pvalue;
        Pvalue.x = 0;
        Pvalue.y = 0;
        Pvalue.z = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++)
            {
                int curRow = by + blurRow;
                int curCol = bx + blurCol;

                Pvalue.x += tile[curRow][curCol].x;
                Pvalue.y += tile[curRow][curCol].y;
                Pvalue.z += tile[curRow][curCol].z;
                pixels++; // keep track of number of pixels in the accumulated total
            }
        }

        // write the new pixel value to output
        out[row * width + col].x = (unsigned char)(Pvalue.x / pixels);
        out[row * width + col].y = (unsigned char)(Pvalue.y / pixels);
        out[row * width + col].z = (unsigned char)(Pvalue.z / pixels);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "format: " << argv[0] << " { 24-bit BMP Image Filename }" << endl;
        exit(1);
    }
    string input_filename(argv[1]);
    string output_filename = "./" + input_filename.substr(0, input_filename.find_last_of(".")) + "_shared_blurred.bmp";

    bitmap_image bmp(argv[1]);

    if (!bmp)
    {
        cerr << "Image not found" << endl;
        exit(1);
    }

    int height = bmp.height();
    int width = bmp.width();
    int image_size = height * width;

    cout << "Image dimensions:" << endl;
    cout << "height: " << height << " width: " << width << endl;

    cout << "Converting " << argv[1] << " from color to grayscale..." << endl;

    // Transform image into vector of doubles
    vector<uchar3> input_image;
    rgb_t color;
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            bmp.get_pixel(x, y, color);
            input_image.push_back({color.red, color.green, color.blue});
        }
    }

    vector<uchar3> output_image(input_image.size());

    uchar3 *d_in, *d_out;
    int img_size = (input_image.size() * sizeof(char) * 3);
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);

    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, input_image.data(), img_size, cudaMemcpyHostToDevice);
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    // start timer
    cudaEventRecord(start, 0);

    // Switched the height and width in the block and kernel call
    //  TODO: Fill in the correnct blockSize and gridSize
    dim3 dimGrid(ceil(height / 16.0) + 1, ceil(width / 16.0) + 1, 1);
    dim3 dimBlock(16, 16, 1);

    blurKernel<<<dimGrid, dimBlock>>>(d_in, d_out, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float diff;
    cudaEventElapsedTime(&diff, start, stop);
    printf("time: %f ms\n", diff);
    std::ofstream output_file("./shared_time.csv", std::ios::out | std::ios::app);
    output_file << diff << "," << image_size << std::endl;
    output_file.close();

    // deallocate timers
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Set updated pixels

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int pos = x * height + y;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    cout << "Conversion complete." << endl;

    bmp.save_image(output_filename);

    cudaFree(d_in);
    cudaFree(d_out);
}
