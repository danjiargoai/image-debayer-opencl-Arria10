#ifndef WIDTH
#define WIDTH 1280
#endif
#ifndef HEIGHT
#define HEIGHT 800
#endif
#define WINDOW_SIZE 5
#ifndef UNROLLTIMES
#define UNROLLTIMES 64
#endif

#define PO(image, x, y, W) ((image)[y*W+x])
#define P(image, x, y, W, H) ((image)[(y >= H ? H - 1 : y < 0 ? 0 : y) * W + (x >= W ? W - 1 : x < 0 ? 0 : x)])
#define PW(image, x, y) P(image, x, y, WIDTH, HEIGHT)

kernel void debayer(global uchar* restrict in, global uchar4* restrict out, int outstride, uchar offset_x, uchar offset_y) {
    // Create shift register
    #define SHIFT_REG_SIZE (WINDOW_SIZE-1)*WIDTH+1
    private uchar im_reg[SHIFT_REG_SIZE];
    
    // Initialize shift register to all 0s
    #pragma unroll
    for (unsigned int i = 0; i < SHIFT_REG_SIZE; ++i) {
        im_reg[i] = 0;
    }

    //for (unsigned int y = 0; y < 2; ++y) {
    //  // Would be accessing WIDTH*8 bits of data
    //  #pragma unroll
    //  for (unsigned int x = 0; x < WIDTH; ++x) {
    //      im_reg[(2 + y) * WIDTH + x] = PO(in, x, y, WIDTH);
    //  }
    //}

    #pragma unroll
    for (unsigned int m = 0; m < 2*WIDTH + 1; ++m) {
        im_reg[m + 2*WIDTH] = in[m];
    }
    const unsigned int start_pos = 2*WIDTH + 1;
    
    const float fs1_a = -0.125f, fs1_b = -0.125f,  fs1_c =  0.0625f, fs1_d = -0.1875f;
    const float fs2_a = -0.125f, fs2_b =  0.0625f, fs2_c = -0.125f , fs2_d = -0.1875f;
    const float fs3_a =  0.25f , fs3_b =  0.5f;
    const float fs4_a =  0.25f , fs4_c =  0.5f;
    const float fs5_b = -0.125f, fs5_c = -0.125f, fs5_d =  0.25f;
    const float fs6_a = 0.5f , fs6_b = 0.625f, fs6_c = 0.625f, fs6_d = 0.75f;

    for (unsigned int j = 0; j < HEIGHT; j += 1) {
        bool j_0 = (j & 0x1) ^ (offset_y & 0x1);
        // Number of operations. Largely depends on device resource
        #pragma unroll UNROLLTIMES
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
