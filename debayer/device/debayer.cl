#ifndef WIDTH
#define WIDTH 1280
#endif
#ifndef HEIGHT
#define HEIGHT 800
#endif
#define WINDOW_SIZE 5
#ifndef UNROLL_DEBAYER
#define UNROLL_DEBAYER 32
#endif
#ifndef UNROLL_RECT
#define UNROLL_RECT 16
#endif

#define PO(image, x, y, W) ((image)[y*W+x])
#define P(image, x, y, W, H) ((image)[(y >= H ? H - 1 : y < 0 ? 0 : y) * W + (x >= W ? W - 1 : x < 0 ? 0 : x)])
#define PW(image, x, y) P(image, x, y, WIDTH, HEIGHT)

__kernel void debayer(__global uchar* restrict in, __global uchar4* restrict out, //input and output
					int outstride, uchar offset_x, uchar offset_y) {
    // Create shift register
    #define SHIFT_REG_SIZE (WINDOW_SIZE-1)*WIDTH+1
    __private uchar im_reg[SHIFT_REG_SIZE];

    // Initialize shift register to all 0s
    #pragma unroll
    for (unsigned int i = 0; i < SHIFT_REG_SIZE; ++i) {
        im_reg[i] = 0;
    }

    #pragma unroll
    for (unsigned int m = 0; m < 2*WIDTH + 1; ++m) {
        im_reg[m + 2*WIDTH] = in[m];
    }
    const unsigned int start_pos = 2*WIDTH + 1;
    
    // Filter values
    const float fs1_a = -0.125f, fs1_b = -0.125f,  fs1_c =  0.0625f, fs1_d = -0.1875f;
    const float fs2_a = -0.125f, fs2_b =  0.0625f, fs2_c = -0.125f , fs2_d = -0.1875f;
    const float fs3_a =  0.25f , fs3_b =  0.5f;
    const float fs4_a =  0.25f , fs4_c =  0.5f;
    const float fs5_b = -0.125f, fs5_c = -0.125f, fs5_d =  0.25f;
    const float fs6_a = 0.5f , fs6_b = 0.625f, fs6_c = 0.625f, fs6_d = 0.75f;

	// Debayer start here
    for (unsigned int j = 0; j < HEIGHT; j += 1) {
        bool j_0 = (j & 0x1) ^ (offset_y & 0x1);
        // Number of operations. Largely depends on device resource
        #pragma unroll UNROLL_DEBAYER
        for (unsigned int i = 0; i < WIDTH; i += 1) {
            #pragma pipeline
            uchar4 dest;
            bool i_0 = (i & 0x1) ^ (offset_x & 0x1);

            // (0,2) is the center location
            // Debayer algorithm starts here
            int r, g, b;
            float s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f; 
            // Somehow MACRO didnt work...
            s1 = (float)(im_reg[2*WIDTH-2] + im_reg[2*WIDTH+2]);
            s2 = (float)(im_reg[0] + im_reg[4*WIDTH]);
            s3 = (float)(im_reg[2*WIDTH - 1] + im_reg[2*WIDTH+1]);
            s4 = (float)(im_reg[WIDTH] + im_reg[3*WIDTH]);
            s5 = (float)(im_reg[WIDTH-1] + im_reg[WIDTH+1] + im_reg[3*WIDTH-1] + im_reg[3*WIDTH + 1]);
            s6 = (float)(im_reg[2*WIDTH]);

            float filter_a = fs1_a*s1 + fs2_a*s2 + fs3_a*s3 + fs4_a*s4 + 0.0f     + fs6_a*s6;
            float filter_b = fs1_b*s1 + fs2_b*s2 + fs3_b*s3 + 0        + fs5_b*s5 + fs6_b*s6;
            float filter_c = fs1_c*s1 + fs2_c*s2 + 0        + fs4_c*s4 + fs5_c*s5 + fs6_c*s6;
            float filter_d = fs1_d*s1 + fs2_d*s2 + 0 + 0 + fs5_d*s5 + fs6_d*s6;

            filter_a = filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0;
            filter_b = filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0;
            filter_c = filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0;
            filter_d = filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0;

            if (!i_0 && !j_0) { // 0,0
                dest.w = (uchar)s6;
                dest.x = (uchar)filter_a;//(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
                dest.y = (uchar)filter_d;//(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
                dest.z = 0;
            }
            else if (i_0 && !j_0) { // 1,0
                dest.x = (uchar)s6;
                dest.w = (uchar)filter_b;//(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
                dest.y = (uchar)filter_c;//(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
                dest.z = 0;
            }
            else if (!i_0 && j_0) { // 0,1
                dest.x = (uchar)s6;
                dest.w = (uchar)filter_c;//(filter_c > 0 ? filter_c > 255.0 ? 255.0 : filter_c : 0);
                dest.y = (uchar)filter_b;//(filter_b > 0 ? filter_b > 255.0 ? 255.0 : filter_b : 0);
                dest.z = 0;
            }
            else { // 1,1
                dest.y = (uchar)s6;
                dest.w = (uchar)filter_d;//(filter_d > 0 ? filter_d > 255.0 ? 255.0 : filter_d : 0);
                dest.x = (uchar)filter_a;//(filter_a > 0 ? filter_a > 255.0 ? 255.0 : filter_a : 0);
                dest.z = 0;
            }
            
            // output indices
            int dest_loc  = j*WIDTH + i;
            out[dest_loc] = dest;
            #pragma unroll
            for (unsigned int i_shift = 0; i_shift < SHIFT_REG_SIZE - 1; ++i_shift) {
                im_reg[i_shift] = im_reg[i_shift + 1];
            }
            im_reg[SHIFT_REG_SIZE-1] = in[start_pos + j*WIDTH + i];
        }
    }
}

// This is still WIP
__kernel void bilinear_rectification(__global uchar4* restrict in, __global uchar4* restrict out,
								   __constant short2* restrict indices, __constant ushort4* restrict weights,
								   int stride) {
	for (unsigned j = 0; j < HEIGHT; j += 1) {
		#pragma unroll UNROLL_RECT
		for (unsigned i = 0; i < WIDTH; i += 1) {
			int index = j * WIDTH + i;
			int offset = indices[index].s1 * stride + indices[index].s0;
			uchar4 pix00 = in[offset];
			uchar4 pix10 = in[offset + 1];
			uchar4 pix01 = in[offset + stride];
			uchar4 pix11 = in[offset + stride + 1];
			ushort4 weight = weights[index];
			out[index].w = pix00.w * weight.w + pix10.w * weight.x + pix01.w * weight.y + pix11.w * weight.z;
			out[index].x = pix00.x * weight.w + pix10.x * weight.x + pix01.x * weight.y + pix11.x * weight.z;
			out[index].y = pix00.y * weight.w + pix10.y * weight.x + pix01.y * weight.y + pix11.y * weight.z;
			out[index].z = 0;
		}
	}
}
