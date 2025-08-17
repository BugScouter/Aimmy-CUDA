using System.Drawing;
using System.Drawing.Imaging;
using static Aimmy2.AILogic.AIManager;

namespace Aimmy2.AILogic
{
    public static class MathUtil
    {

        public static Func<double[], double[], double> L2Norm_Squared_Double = (x, y) =>
        {
            double dist = 0f;
            for (int i = 0; i < x.Length; i++)
            {
                dist += (x[i] - y[i]) * (x[i] - y[i]);
            }

            return dist;
        };
        public static float Distance(Prediction a, Prediction b)
        {
            float dx = a.ScreenCenterX - b.ScreenCenterX;
            float dy = a.ScreenCenterY - b.ScreenCenterY;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        public static unsafe void BitmapToFloatArrayInPlace(Bitmap image, float[] result, int IMAGE_SIZE)
        {
            int width = IMAGE_SIZE;
            int height = IMAGE_SIZE;
            int totalPixels = width * height;
            const float multiplier = 1f / 255f;

            var rect = new Rectangle(0, 0, width, height);
            var bmpData = image.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb); //3 bytes per pixel
                                                                                                    // blue green red (strict order)
            try
            {
                int stride = bmpData.Stride;
                byte* basePtr = (byte*)bmpData.Scan0;
                int redOffset = 0;
                int greenOffset = totalPixels;
                int blueOffset = 2 * totalPixels;

                //for each row in the image -> create a temporary array for red green and blue (rgb)
                Parallel.For(0, height, () => (localR: new float[width], localG: new float[width], localB: new float[width]),
                (y, state, local) =>
                {
                    byte* row = basePtr + (y * stride);

                    // process entire row in local buffers
                    for (int x = 0; x < width; x++)
                    {
                        int bufferIndex = x * 3;
                        // BGR byte order: +2 = R, +1 = G, +0 = B
                        // B = 0
                        // G = 1
                        // R = 2
                        // (bufferIndex + x)
                        local.localR[x] = row[bufferIndex + 2] * multiplier;
                        local.localG[x] = row[bufferIndex + 1] * multiplier;
                        local.localB[x] = row[bufferIndex] * multiplier;
                    }

                    // after processing the row copy the results into the final array
                    int rowStart = y * width;
                    Array.Copy(local.localR, 0, result, redOffset + rowStart, width);
                    Array.Copy(local.localG, 0, result, greenOffset + rowStart, width);
                    Array.Copy(local.localB, 0, result, blueOffset + rowStart, width);

                    return local;
                },
                _ => { });
            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }
    }
}
