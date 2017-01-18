#ifndef _REF_H_
#define _REF_H_
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>

#define PO(image, x, y, W) ((image)[y*W + x])
#define P(image, x, y, W, H) ((image)[( (y)>=H ? H-1 : ((y)<0 ? 0:(y)) )*(W)+(             (x)>=W ? W-1 : ((x)<0 ? 0 : (x)) )])
#define PW(image, x, y) P(image, x, y, WIDTH, HEIGHT)

const int WIDTH = 1280;
const int HEIGHT = 800;
const int STRIDE = 1280;
const int NUM_PIXELS = WIDTH*HEIGHT;
typedef cl_uchar uchar;

void debayer_bggr_to_u8xbpp_hqlinear(const uint8_t* const in, cl_uchar4* const out,
                                            const int width, const int height,
                                            const int instride, const int outstride, uchar offset_x, uchar offset_y);

#endif
