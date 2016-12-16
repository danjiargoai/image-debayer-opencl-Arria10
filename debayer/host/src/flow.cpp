#include <iostream>
#include <math.h>
#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include "utility.h"

const char image_file1[] = "input1.raw";
const int WIDTH = 1280;
const int HEIGHT = 800;
const int STRIDE = 1280;
const int NUM_PIXELS = WIDTH*HEIGHT;

typedef cl_uchar uchar;

#define PO(image, x, y, W) ((image)[y*W + x])
#define P(image, x, y, W, H) ((image)[( (y)>=H ? H-1 : ((y)<0 ? 0:(y)) )*(W)+(             (x)>=W ? W-1 : ((x)<0 ? 0 : (x)) )])
#define PW(image, x, y) P(image, x, y, WIDTH, HEIGHT)

bool init_platform();
bool run();
bool verify();
void cleanup();
bool save_image(const char *outf, uchar *image1, int width, int height, char grayscale);
bool read_image_raw(const char *outf, uchar *image1, int width, int height);
void debayer_bggr_to_u8xbpp_hqlinear(const uint8_t* const in, cl_uchar4* const out,
                                            const int width, const int height,
                                            const int instride, const int outstride, uchar offset_x, uchar offset_y);

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

void debayer_bggr_to_u8xbpp_hqlinear(const uint8_t* const in, cl_uchar4* const dest,
                                            const int width, const int height,
                                            const int instride, const int outstride, uchar offset_x, uchar offset_y)
{
	const float fs1_a = -0.125f, fs1_b = -0.125f,  fs1_c =  0.0625f, fs1_d = -0.1875f;
	const float fs2_a = -0.125f, fs2_b =  0.0625f, fs2_c = -0.125f , fs2_d = -0.1875f;
	const float fs3_a =  0.25f , fs3_b =  0.5f;
	const float fs4_a =  0.25f , fs4_c =  0.5f;
	const float fs5_b = -0.125f, fs5_c = -0.125f, fs5_d =  0.25f;
	const float fs6_a = 0.5f , fs6_b = 0.625f, fs6_c = 0.625f, fs6_d = 0.75f;

	for (unsigned int j = 2; j < height-2; j += 1) {
		// Number of operations. Largely depends on device resource
		#pragma unroll 4
		for (unsigned int i = 2; i < width-2; i += 1) {
			// (x,y) is the center location
			int x = i, y = j;
			// Debayer algorithm starts here
			int r, g, b;
			float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f; 
			s1 = PW(in, x-2, y) + PW(in, x+2, y);
			s2 = PW(in, x, y-2) + PW(in, x, y+2);
			s3 = PW(in, x-1, y) + PW(in, x+1, y);
			s4 = PW(in, x, y-1) + PW(in, x, y+1);
			s5 = PW(in, x-1, y-1) + PW(in, x+1, y-1) + PW(in, x-1, y+1) + PW(in, x+1, y+1);
			s6 = PW(in, x, y);
			
			float filter_a = fs1_a*s1 + fs2_a*s2 + fs3_a*s3 + fs4_a*s4 + 0.0f     + fs6_a*s6;
			float filter_b = fs1_b*s1 + fs2_b*s2 + fs3_b*s3 + 0        + fs5_b*s5 + fs6_b*s6;
			float filter_c = fs1_c*s1 + fs2_c*s2 + 0        + fs4_c*s4 + fs5_c*s5 + fs6_c*s6;
			float filter_d = fs1_d*s1 + fs2_d*s2 + 0 + 0 + fs5_d*s5 + fs6_d*s6;
			int dest_w = j * width + i;

			if (((x + offset_x) % 2 == 0) && ((y + offset_y) % 2 == 0)) { // 0,0
				dest[dest_w].w = (uchar)s6;
				dest[dest_w].x = (uchar)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
				dest[dest_w].y = (uchar)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
				dest[dest_w].z = 0;
			}
			else if (((x + offset_x) % 2 == 1) && ((y + offset_y) % 2 == 0)) { // 1,0
				dest[dest_w].x = (uchar)s6;
				dest[dest_w].w = (uchar)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
				dest[dest_w].y = (uchar)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
				dest[dest_w].z = 0;
			}
			else if (((x + offset_x) % 2 == 0) && ((y + offset_y) % 2 == 1)) { // 0,1
				dest[dest_w].x = (uchar)s6;
				dest[dest_w].w = (uchar)(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
				dest[dest_w].y = (uchar)(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
				dest[dest_w].z = 0;
			}
			else { // 1,1
				dest[dest_w].y = (uchar)s6;
				dest[dest_w].w = (uchar)(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
				dest[dest_w].x = (uchar)(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
				dest[dest_w].z = 0;
			}
		}
	}
}
