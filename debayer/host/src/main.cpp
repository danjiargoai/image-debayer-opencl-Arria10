#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include "utility.h"
#include "ref.h"

const char image_file1[] = "input1.raw";

bool init_platform();
bool run();
bool verify();
void cleanup();
bool save_image(const char *outf, uchar *image1, int width, int height, char grayscale);
bool read_image_raw(const char *outf, uchar *image1, int width, int height);
cl_uchar3* convert_uchar4(uchar* a) {
	cl_uchar3* output = (cl_uchar3*)malloc(sizeof(a));
	for (uint32_t i = 0; i < sizeof(a); ++i) {
		if (i % 3 == 0)
			output[i/3].w = a[i];
		else if (i % 3 == 1)
			output[i/3].x = a[i];
		else
			output[i/3].y = a[i];
	}
	
	free(a);
	return output;
}

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_program program;
static cl_int status;
static cl_command_queue queue;

static cl_mem src, dst;
static uchar* v_src = 0;
static cl_uchar4* v_dst = 0;

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
        for (int i = 0; i < width; ++i)
            fscanf(input, "%hhu ", &(image1[j*width+i]));
        fscanf(input, "\n");
    }
    fclose(input);
    return true;
}

bool run() {
    printf("Creating write Buffers\n");
    src = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, NUM_PIXELS, NULL, &status);
    assert(status == CL_SUCCESS);
    v_src = (uchar *)clEnqueueMapBuffer(queue, src, CL_TRUE, CL_MAP_READ, 0, NUM_PIXELS, 0, NULL, NULL, &status);
    assert(status == CL_SUCCESS);

    dst = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, WIDTH*HEIGHT*4, NULL, &status);
    assert(status == CL_SUCCESS);
    v_dst = (cl_uchar4*)clEnqueueMapBuffer(queue, dst, CL_TRUE, CL_MAP_WRITE, 0, WIDTH*HEIGHT*4, 0, NULL, NULL, &status);
    assert(status == CL_SUCCESS);
    
    printf("Reading input images\n");
    if (!read_image_raw(image_file1, v_src, WIDTH, HEIGHT)) return false;
    
    printf("Initializing kernels\n");
    size_t task_dim = 1;
    cl_kernel kernel = clCreateKernel(program, "debayer", &status);
    assert(status == CL_SUCCESS);
    
    cl_event kernel_event;
	
	cl_uint outstride = STRIDE*3;
	cl_uchar offset_x = 0, offset_y = 0;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &src);
    assert(status == CL_SUCCESS);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 2, sizeof(cl_uint), &outstride);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 3, sizeof(cl_uchar), &offset_x);
	assert(status == CL_SUCCESS);
	status = clSetKernelArg(kernel, 4, sizeof(cl_uchar), &offset_y);
	assert(status == CL_SUCCESS);
    status = clEnqueueTask(queue, kernel, 0, NULL, &kernel_event);
    assert(status == CL_SUCCESS);
    
    clWaitForEvents(1, &kernel_event);
    status = clFlush(queue);
	assert(status == CL_SUCCESS);
    
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
		clEnqueueUnmapMemObject (queue, dst, v_dst, 0, NULL, NULL);
		clReleaseMemObject (dst);
		v_dst = 0;
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

