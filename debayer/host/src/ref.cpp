#include "ref.h"
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
