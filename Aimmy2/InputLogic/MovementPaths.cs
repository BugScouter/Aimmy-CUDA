using System.Drawing;

namespace InputLogic
{
    class MovementPaths
    {
        private static readonly int[] permutation = new int[512];

        internal static Point CubicBezier(Point start, Point end, Point control1, Point control2, double t)
        {
            double u = 1 - t;
            double tt = t * t;
            double uu = u * u;

            double x = uu * u * start.X + 3 * uu * t * control1.X + 3 * u * tt * control2.X + tt * t * end.X;
            double y = uu * u * start.Y + 3 * uu * t * control1.Y + 3 * u * tt * control2.Y + tt * t * end.Y;

            return new Point((int)x, (int)y);
        }

        internal static Point Lerp(Point start, Point end, double t)
        {
            int x = (int)(start.X + (end.X - start.X) * t);
            int y = (int)(start.Y + (end.Y - start.Y) * t);
            return new Point(x, y);
        }


        internal static Point Exponential(Point start, Point end, double t, double exponent = 2.0)
        {
            double factor = Math.Pow(t, exponent);
            double x = start.X + (end.X - start.X) * factor;
            double y = start.Y + (end.Y - start.Y) * factor;
            return new Point((int) x, (int) y);
        }

        internal static Point Adaptive(Point start, Point end, double t, double threshold = 100.0)
        {
            int dx = end.X - start.X;
            int dy = end.Y - start.Y;
            double distanceSq = dx * dx + dy * dy;

            //double distance = Math.Sqrt(Math.Pow(end.X - start.X, 2) + Math.Pow(end.Y - start.Y, 2));
            if (distanceSq < threshold * threshold)
            {
                return Lerp(start, end, t);
            }
            else
            {
                Point control1 = new Point(start.X + dx / 3, start.Y + dy / 3);
                Point control2 = new Point(start.X + 2 * dx / 3, start.Y + 2 * dy / 3);

                return CubicBezier(start, end, control1, control2, t);
            }
        }

        internal static Point PerlinNoise(Point start, Point end, double t, double amplitude = 10.0, double frequency = 0.1)
        {
            double dx = end.X - start.X;
            double dy = end.Y - start.Y;

            double baseX = start.X + dx * t;
            double baseY = start.Y + dy * t;

            double noiseX = Noise(t * frequency, 0) * amplitude;
            double noiseY = Noise(t * frequency, 100) * amplitude;

            double perpX = -dy;
            double perpY = dx;
            double perpLength = Math.Sqrt(perpX * perpX + perpY * perpY);

            if (perpLength > 1e-6)
            {
                perpX /= perpLength;
                perpY /= perpLength;
            }

            double finalX = baseX + perpX * noiseX + noiseY * 0.3;
            double finalY = baseY + perpY * noiseX + noiseY * 0.3;

            return new Point((int)finalX, (int)finalY);
        }

        private static double Fade(double t) => t * t * t * (t * (t * 6 - 15) + 10);

        private static double Lerp(double a, double b, double t) => a + t * (b - a);

        private static double Grad(int hash, double x, double y)
        {
            int h = hash & 15;
            double u = h < 8 ? x : y;
            double v = h < 4 ? y : h == 12 || h == 14 ? x : 0;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }

        private static double Noise(double x, double y)
        {
            int X = (int)Math.Floor(x) & 255;
            int Y = (int)Math.Floor(y) & 255;

            x -= Math.Floor(x);
            y -= Math.Floor(y);

            double u = Fade(x);
            double v = Fade(y);

            int A = permutation[X] + Y;
            int AA = permutation[A];
            int AB = permutation[A + 1];
            int B = permutation[X + 1] + Y;
            int BA = permutation[B];
            int BB = permutation[B + 1];

            return Lerp(Lerp(Grad(permutation[AA], x, y),
                           Grad(permutation[BA], x - 1, y), u),
                      Lerp(Grad(permutation[AB], x, y - 1),
                           Grad(permutation[BB], x - 1, y - 1), u), v);
        }
    }
}