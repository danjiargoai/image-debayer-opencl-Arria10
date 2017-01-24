#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include "utility.h"
#include "ref.h"
#define AOCL_ALIGNMENT 64

const char image_file1[] = "input1.raw";

bool init_platform();
bool run();
bool verify();
void cleanup();
bool save_image(const char *outf, uchar *image1, int width, int height, char grayscale);
bool read_image_raw(const char *outf, uchar *image1, int width, int height);
bool read_indices(cl_ushort2 *indices);
bool read_weights(cl_ushort4 *weights);

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_program program;
static cl_int status;
static cl_command_queue queue;

static cl_mem src, dst0, dst1, indices, weights;
static uchar* v_src = 0;
static cl_uchar4* v_dst = 0;
static cl_ushort2* v_indices = 0;
static cl_ushort4* v_weights = 0;

int main(int argc, char** argv) {
    if (!init_platform()) return 1;
    if (!run()) return 1;
    if (!verify()) return 1;
    cleanup();
    return 0;
}

bool save_image_ppm (const char *outf, cl_uchar3 *image1, int width, int height) {
    FILE *output = fopen(outf, "wb");
    if (output == NULL) {
        printf("Couldn't open %s for writing!\n", outf);
        return false;
    }

    fprintf(output, "P6\n%d %d\n%d\n", width, height, 255);
    for (int j = 0; j < height; ++j) 
        for (int i = 0; i < width; ++i)
            fwrite(&image1[j*width+i], 1, 3, output);
    fclose(output);
    return true;
}

bool read_image_raw(const char* outf, uchar* image1, int width, int height) {
    FILE *input = fopen(outf, "r");
    if (input == NULL) {
        printf("Couldn't open %s for reading!\n", outf);
        return false;
    }

    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            fscanf(input, "%hhu ", &(image1[j*width+i]));
        }
        fscanf(input, "\n");
    }
    fclose(input);
    return true;
}

bool read_indices(cl_ushort2* indices) {
    for (unsigned j = 0; j < HEIGHT; j += 1)
        for (unsigned i = 0; i < WIDTH; i += 1) {
            indices[j * WIDTH + i].s0 = i;
            indices[j * WIDTH + i].s1 = j;
        }
    return true;
}

bool read_weights(cl_ushort4* weights) {
    for (unsigned j = 0; j < HEIGHT; j += 1)
        for (unsigned i = 0; i < WIDTH; i += 1) {
            weights[j * WIDTH + i].w = 1;
            weights[j * WIDTH + i].x = 0;
            weights[j * WIDTH + i].y = 0;
            weights[j * WIDTH + i].z = 0;
        }
    return true;
}

bool run() {
    unsigned long starttime, endtime;
    void *ptr_s = NULL;
    void *ptr_d = NULL;
    void *ptr_indices = NULL;
    void *ptr_weights = NULL;
    posix_memalign(&ptr_s, AOCL_ALIGNMENT, NUM_PIXELS);
    posix_memalign(&ptr_d, AOCL_ALIGNMENT, WIDTH*HEIGHT*4);
    posix_memalign(&ptr_indices, AOCL_ALIGNMENT, WIDTH*HEIGHT*4);
    posix_memalign(&ptr_weights, AOCL_ALIGNMENT, WIDTH*HEIGHT*8);
    
    v_src = (uchar*)ptr_s;
    v_dst = (cl_uchar4*)ptr_d;
    v_indices = (cl_ushort2*)ptr_indices;
    v_weights = (cl_ushort4*)ptr_weights;
    
    printf("\nReading input images\n");
    if (!read_image_raw(image_file1, v_src, WIDTH, HEIGHT)) return false;
    
    printf("\nReading indices\n");
    if (!read_indices(v_indices)) return false;
    
    printf("\nReading weights\n");
    if (!read_weights(v_weights)) return false;
    
    printf("\nCreating Buffers\n");
    src = clCreateBuffer(context, CL_MEM_READ_ONLY, NUM_PIXELS, NULL, &status);
    assert(status == CL_SUCCESS);
    
    dst0 = clCreateBuffer(context, CL_MEM_READ_WRITE, WIDTH*HEIGHT*4, NULL, &status);
    assert(status == CL_SUCCESS);
    
    dst1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, WIDTH*HEIGHT*4, NULL, &status);
    assert(status == CL_SUCCESS);
    
    indices = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*4, NULL, &status);
    assert(status == CL_SUCCESS);

    weights = clCreateBuffer(context, CL_MEM_READ_ONLY, WIDTH*HEIGHT*8, NULL, &status);
    assert(status == CL_SUCCESS);

    printf("\nWrite Buffer\n");
    cl_event cl_src_map_buffer;
    status = clEnqueueWriteBuffer(queue, src, CL_TRUE, 0, NUM_PIXELS, v_src, 0, NULL, &cl_src_map_buffer);
    assert(status == CL_SUCCESS);

    cl_event cl_indices_map_buffer;
    status = clEnqueueWriteBuffer(queue, indices, CL_TRUE, 0, WIDTH * HEIGHT * 4, v_indices, 0, NULL, &cl_indices_map_buffer);
    assert(status == CL_SUCCESS);
    
    cl_event cl_weights_map_buffer;
    status = clEnqueueWriteBuffer(queue, weights, CL_TRUE, 0, WIDTH * HEIGHT * 8, v_weights, 0, NULL, &cl_weights_map_buffer);
    assert(status == CL_SUCCESS);

    printf("\nInitializing kernels\n");
    size_t task_dim = 1;
    cl_kernel debayer_kernel = clCreateKernel(program, "debayer", &status);
    assert(status == CL_SUCCESS);
    cl_kernel bilinear_rect_kernel = clCreateKernel(program, "bilinear_rectification", &status);
    assert(status == CL_SUCCESS);
    
    cl_event kernel_event[2];
    
    // Debayer kernel arguments
    cl_uint outstride = STRIDE*3;
    cl_uchar offset_x = 0, offset_y = 0;
    status = clSetKernelArg(debayer_kernel, 0, sizeof(cl_mem), &src);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(debayer_kernel, 1, sizeof(cl_mem), &dst0);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(debayer_kernel, 2, sizeof(cl_uint), &outstride);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(debayer_kernel, 3, sizeof(cl_uchar), &offset_x);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(debayer_kernel, 4, sizeof(cl_uchar), &offset_y);
    assert(status == CL_SUCCESS);
    status = clEnqueueTask(queue, debayer_kernel, 0, NULL, &kernel_event[0]);
    assert(status == CL_SUCCESS);
    
    // Bilinear rectification kernel arguments
    status = clSetKernelArg(bilinear_rect_kernel, 0, sizeof(cl_mem), &dst0);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(bilinear_rect_kernel, 1, sizeof(cl_mem), &dst1);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(bilinear_rect_kernel, 2, sizeof(cl_mem), &indices);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(bilinear_rect_kernel, 3, sizeof(cl_mem), &weights);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(bilinear_rect_kernel, 4, sizeof(cl_uint), &STRIDE);
    assert(status == CL_SUCCESS);
    status = clEnqueueTask(queue, bilinear_rect_kernel, 0, NULL, &kernel_event[1]);
    assert(status == CL_SUCCESS);
    
    clWaitForEvents(1, kernel_event);
    status = clFlush(queue);
    assert(status == CL_SUCCESS);
    
    cl_event cl_dst_map_buffer;
    status = clEnqueueReadBuffer(queue, dst1, CL_TRUE, 0, WIDTH * HEIGHT*4, v_dst, 0, NULL, &cl_dst_map_buffer);
    assert(status == CL_SUCCESS);
    clFinish(queue);
    clWaitForEvents(1, &cl_dst_map_buffer);
    status = clGetEventProfilingInfo(cl_src_map_buffer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
    assert(status == CL_SUCCESS);
    status = clGetEventProfilingInfo(cl_src_map_buffer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
    assert(status == CL_SUCCESS);

    unsigned long elapsed = endtime - starttime;
    printf("\nHost to Device time is %0.3f ms\n", (elapsed / 1000000.0));

    status = clGetEventProfilingInfo(kernel_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
    assert(status == CL_SUCCESS);
    status = clGetEventProfilingInfo(kernel_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
    assert(status == CL_SUCCESS);

    elapsed = endtime - starttime;
    printf("\nKernel execution time is %0.3f ms\n", (elapsed / 1000000.0));

    status = clGetEventProfilingInfo(cl_dst_map_buffer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL);
    assert(status == CL_SUCCESS);
    status = clGetEventProfilingInfo(cl_dst_map_buffer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL);
    assert(status == CL_SUCCESS);

    elapsed = endtime - starttime;
    printf("\nDevice to Host time is %0.3f ms\n", (elapsed / 1000000.0));

    printf("Save output image.\n");
    if(!save_image_ppm ("out.ppm", v_dst, STRIDE, HEIGHT)) return false;
}

bool init_platform() {
    cl_uint num_platforms;
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    printf("Number of platform found %d\n", num_platforms);
    assert(status == CL_SUCCESS);
    
    status = clGetPlatformIDs(1, &platform, NULL); 
    if (platform == NULL) {
        printf("ERROR: Unable to find platform.\n");
        return false;
    }
    
    cl_uint num_devices;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    printf("Number of devices found %d\n", num_devices);
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    context = clCreateContext(0, 1, &device, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    assert(status == CL_SUCCESS);

    size_t length;
    unsigned char* binary_file = get_binary("debayer.aocx", &length);
    program = clCreateProgramWithBinary(context, 1, &device, &length, (const unsigned char **)&binary_file, NULL, &status);
    assert(status == CL_SUCCESS);

    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    assert(status == CL_SUCCESS);

    return true;
}

void cleanup() {
    if (v_src) {
        clEnqueueUnmapMemObject (queue, src, v_src, 0, NULL, NULL);
        clReleaseMemObject (src);
        v_src = 0;
    }
    if (v_dst) {
        clEnqueueUnmapMemObject (queue, dst1, v_dst, 0, NULL, NULL);
        clReleaseMemObject (dst1);
        v_dst = 0;
    }
    if (v_indices) {
        clEnqueueUnmapMemObject (queue, indices, v_indices, 0, NULL, NULL);
        clReleaseMemObject (indices);
        v_indices = 0;
    }
    if (v_weights) {
        clEnqueueUnmapMemObject (queue, weights, v_weights, 0, NULL, NULL);
        clReleaseMemObject (weights);
        v_weights = 0;
    }
    if(program) {
        clReleaseProgram(program);
    }
    if(queue) {
        clReleaseCommandQueue(queue);
    }
    if(context) {
        clReleaseContext(context);
    }
}

bool verify() {
    printf("Start Verifying\n");
    cl_uchar4* output = (cl_uchar4*)malloc(WIDTH*4*HEIGHT);
    debayer_bggr_to_u8xbpp_hqlinear((const uint8_t*)v_src, output, WIDTH, HEIGHT, WIDTH, WIDTH*3, 0, 0);
    
    printf("Correctness check\n");
    for (uint32_t j = 2; j < HEIGHT-2; ++j) {
        for (uint32_t i = 2; i < WIDTH-2; ++i) {
            int idx = j * WIDTH + i;
            if (output[idx].w != v_dst[idx].w) {
                printf("Result not equal R %d %d: opencl output %d, reference output %d\n", i, j, v_dst[idx].w, output[idx].w);
                free(output);
                return 1;
            }
            if (output[idx].x != v_dst[idx].x) {
                printf("Result not equal G %d %d: opencl output %d, reference output %d\n", i, j, v_dst[idx].x, output[idx].x);
                free(output);
                return 1;
            }
            if (output[idx].y != v_dst[idx].y) {
                printf("Result not equal B %d %d: opencl output %d, reference output %d\n", i, j, v_dst[idx].y, output[idx].y);
                free(output);
                return 1;
            }
        }
    }
    
    printf("VERIFICATION PASSED!!!\n");
    
    free(output);
    return 0;
}

