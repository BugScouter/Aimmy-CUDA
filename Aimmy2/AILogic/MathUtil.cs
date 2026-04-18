using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;
using System.Numerics;

namespace Aimmy2.AILogic
{
    public static class MathUtil
    {
        // bit operations
        private const uint SignMask32 = 0x80000000u;
        private const uint ExpMask32 = 0x7F800000u;
        private const uint MantMask32 = 0x007FFFFFu;

        private const uint SignMask16 = 0x8000u;
        private const uint ExpMask16 = 0x7C00u;
        private const uint MantMask16 = 0x03FFu;

        public static int CalculateNumDetections(int imageSize)
        {
            // YOLOv8 detection calculation: (size/8)² + (size/16)² + (size/32)²
            int stride8 = imageSize / 8;
            int stride16 = imageSize / 16;
            int stride32 = imageSize / 32;

            return (stride8 * stride8) + (stride16 * stride16) + (stride32 * stride32);
        }
        public static Func<double[], double[], double> L2Norm_Squared_Double = (x, y) =>
        {
            double dist = 0f;
            for (int i = 0; i < x.Length; i++)
            {
                dist += (x[i] - y[i]) * (x[i] - y[i]);
            }

            return dist;
        };
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Distance(Prediction a, Prediction b)
        {
            float dx = a.ScreenCenterX - b.ScreenCenterX;
            float dy = a.ScreenCenterY - b.ScreenCenterY;
            return dx * dx + dy * dy;
        }

        // LUT = look up table
        // REFERENCE: https://stackoverflow.com/questions/1089235/where-can-i-find-a-byte-to-float-lookup-table
        // "In this case, the lookup table should be faster than using direct calculation. The more complex the math (trigonometry, etc.), the bigger the performance gain."
        // although we used small calculations, something is better than nothing.
        private static readonly float[] _byteToFloatLut = CreateByteToFloatLut();
        private static float[] CreateByteToFloatLut()
        {
            var lut = new float[256];
            for (int i = 0; i < 256; i++)
                lut[i] = i / 255f;
            return lut;
        }

        /// <summary>
        /// Highly optimized normalization from a raw DirectX / GDI buffer pointer directly into the float array.
        /// This avoids Bitmap creation and extra copies.
        /// </summary>
        public static unsafe void NormalizeDirect(byte* srcPtr, int srcStride, float[] result, int IMAGE_SIZE, int bytesPerPixel = 4)
        {
            int width = IMAGE_SIZE;
            int height = IMAGE_SIZE;
            int totalPixels = width * height;

            // array offsets for RGB planar format
            int rOffset = 0;
            int gOffset = totalPixels;
            int bOffset = totalPixels * 2;

            fixed (float* dest = result)
            {
                float* rPtr = dest + rOffset;
                float* gPtr = dest + gOffset;
                float* bPtr = dest + bOffset;

                // Process rows
                // We avoid Parallel.For for small images to reduce overhead, but keep it for large ones
                if (height >= 320)
                {
                    Parallel.For(0, height, (y) =>
                    {
                        byte* row = srcPtr + (long)y * srcStride;
                        int rowStart = y * width;
                        for (int x = 0; x < width; x++)
                        {
                            int idx = rowStart + x;
                            byte* p = row + (x * bytesPerPixel);
                            
                            // Windows BGR -> Planar RGB (or BGR depending on model, usually BGR for YOLO)
                            // Aimmy's AIManager uses rPtr/gPtr/bPtr mapped to BGR internally
                            bPtr[idx] = _byteToFloatLut[p[0]];
                            gPtr[idx] = _byteToFloatLut[p[1]];
                            rPtr[idx] = _byteToFloatLut[p[2]];
                        }
                    });
                }
                else
                {
                    for (int y = 0; y < height; y++)
                    {
                        byte* row = srcPtr + (long)y * srcStride;
                        int rowStart = y * width;
                        for (int x = 0; x < width; x++)
                        {
                            int idx = rowStart + x;
                            byte* p = row + (x * bytesPerPixel);
                            bPtr[idx] = _byteToFloatLut[p[0]];
                            gPtr[idx] = _byteToFloatLut[p[1]];
                            rPtr[idx] = _byteToFloatLut[p[2]];
                        }
                    }
                }
            }
        }

        // this new function reduces gc pressure as i stopped using array.copy
        // REFERENCE: https://www.codeproject.com/Articles/617613/Fast-Pixel-Operations-in-NET-With-and-Without-unsa
        public static unsafe void BitmapToFloatArrayInPlace(Bitmap image, float[] result, int IMAGE_SIZE)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (result == null) throw new ArgumentNullException(nameof(result));

            int width = IMAGE_SIZE;
            int height = IMAGE_SIZE;
            var rect = new Rectangle(0, 0, width, height);

            var bmpData = image.LockBits(rect, ImageLockMode.ReadOnly, image.PixelFormat);
            try
            {
                NormalizeDirect((byte*)bmpData.Scan0, bmpData.Stride, result, IMAGE_SIZE, 4);
            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }

        #region Half-precision float conversion
        // references: https://devblogs.microsoft.com/dotnet/introducing-the-half-type/
        // https://stackoverflow.com/questions/76799117/how-to-convert-a-float-to-a-half-type-and-the-other-way-around-in-c (in C)
        // https://stackoverflow.com/questions/3026441/float32-to-float16
        // I would just like to say now, that python users are extremely lucky: https://onnxruntime.ai/docs/performance/model-optimizations/float16.html


        /// <summary>
        /// convert single-precision (32-bit) float to half-precision (16-bit) float stored in ushort
        /// </summary>
        /// <param name="f"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort FloatToHalfBits(float f)
        {
            uint fbits = BitConverter.SingleToUInt32Bits(f);
            uint sign = (fbits >> 16) & SignMask16;
            uint val = fbits & ~SignMask32;

            // exactly equal means infinity otherwise its NaN
            // NaN / Inf
            if (val >= ExpMask32)
            {
                if (val == ExpMask32)
                    return (ushort)(sign | ExpMask16); // Inf

                // NaN: preserve top mantissa bits
                return (ushort)(sign | ExpMask16 | ((fbits & MantMask32) >> 13));
            }

            // Too small for normalized half
            if (val < 0x38800000u) // (113 << 23) 
            {
                // subnormal half
                uint mant = (fbits & MantMask32) | (1u << 23);
                int shift = (113 - (int)(fbits >> 23));

                if (shift < 24)
                {
                    uint res = (mant + (1u << (shift - 1)) + ((mant >> shift) & 1u)) >> shift;
                    return (ushort)(sign | (res & MantMask16));
                }

                return (ushort)sign; // underflow to zero
            }

            // Normalized
            uint exp = ((fbits >> 23) - 127 + 15) & 0x1Fu;
            uint mantissa = (fbits >> 13) & MantMask16;
            return (ushort)(sign | (exp << 10) | mantissa); // store as 16 bit
        }

        /// <summary>
        ///  convert half-precision (16-bit) float stored in ushort to single-precision (32-bit) float
        /// </summary>
        /// <param name="h"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float HalfBitsToFloat(ushort h)
        {
            uint sign = (uint)(h & SignMask16) << 16;
            uint exp = (uint)(h & ExpMask16) >> 10;
            uint mant = (uint)(h & MantMask16);

            uint fbits;

            if (exp == 0)
            {
                if (mant == 0)
                {
                    fbits = sign; // Zero
                }
                else
                {
                    // normalize with bit scan
                    int shift = BitOperations.LeadingZeroCount(mant) - 21; // adjust to align to float mantissa
                    mant <<= shift;
                    uint exp32 = (uint)(127 - 14 - shift); // 127 bias - 15 bias + 1
                    fbits = sign | (exp32 << 23) | ((mant & MantMask16) << 13);
                }
            }
            else if (exp == 0x1F)
            {
                // Inf or NaN
                fbits = sign | ExpMask32 | (mant << 13);
            }
            else
            {
                uint exp32 = exp - 15 + 127;
                fbits = sign | (exp32 << 23) | (mant << 13);
            }

            return BitConverter.UInt32BitsToSingle(fbits);
        }
        #endregion

    }
}
