using Aimmy2.Class;
using Other;
using SharpGen.Runtime;
using System.Drawing;
using System.Drawing.Imaging;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using Vortice.Mathematics;
using LogLevel = Other.LogManager.LogLevel;

namespace AILogic
{
    public struct DirectXPointer
    {
        public IntPtr Pointer;
        public int Stride;
        public bool IsValid => Pointer != IntPtr.Zero;
    }

    internal class CaptureManager
    {
        #region Variables
        private string _currentCaptureMethod = ""; // Track current method
        private bool _directXFailedPermanently = false; // Track if DirectX failed with unsupported error
        private bool _notificationShown = false; // Prevent spam notifications

        // Capturing
        public Bitmap? screenCaptureBitmap { get; private set; }
        public Bitmap? directXBitmap { get; private set; }
        private ID3D11Device? _dxDevice;
        private IDXGIOutputDuplication? _deskDuplication;
        private ID3D11Texture2D? _stagingTex;

        // Frame caching for DirectX
        private Bitmap? _cachedFrame;
        private Rectangle _cachedFrameBounds;
        private DateTime _lastFrameTime = DateTime.MinValue;
        private readonly TimeSpan _frameCacheTimeout = TimeSpan.FromMilliseconds(15);


        // Display change handling
        public readonly object _displayLock = new();
        public bool _displayChangesPending { get; set; } = false;

        // Performance tracking
        private int _consecutiveFailures = 0;
        private const int MAX_CONSECUTIVE_FAILURES = 5;

        // stride matching
        private bool _lastStrideMatch = true;
        private int _lastSrcStride = 0;
        private int _lastDstStride = 0;

        #endregion
        #region Handlers
        public CaptureManager()
        {
            // Subscribe to display changes FIRST
            DisplayManager.DisplayChanged += OnDisplayChanged;
        }

        private void OnDisplayChanged(object? sender, DisplayChangedEventArgs e)
        {
            lock (_displayLock)
            {
                _displayChangesPending = true;
                _consecutiveFailures = 0;
                DisposeDxgiResources();
            }
            LogManager.Log(LogLevel.Info, "Display change detected. DirectX resources will be reinitialized.");
        }

        public void HandlePendingDisplayChanges()
        {
            lock (_displayLock)
            {
                if (!_displayChangesPending) return;

                try
                {
                    InitializeDxgiDuplication();
                    _displayChangesPending = false;
                }
                catch (Exception)
                {
                    // Error is logged in InitializeDxgiDuplication
                }
            }
        }

        #endregion
        #region DirectX
        public void InitializeDxgiDuplication()
        {
            DisposeDxgiResources();
            try
            {
                var currentDisplay = DisplayManager.CurrentDisplay;
                if (currentDisplay == null)
                {
                    LogManager.Log(LogLevel.Error, "No current display available. DisplayManager may not be initialized.");
                    throw new InvalidOperationException("No current display available. DisplayManager may not be initialized.");
                }

                using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory1>();
                IDXGIOutput1? targetOutput1 = null;
                IDXGIAdapter1? targetAdapter = null;
                bool foundTarget = false;

                for (uint adapterIndex = 0;
                    factory.EnumAdapters1(adapterIndex, out var adapter).Success;
                    adapterIndex++)
                {
                    LogManager.Log(LogLevel.Info, $"Checking Adapter {adapterIndex}: {adapter.Description.Description.TrimEnd('\0')}");

                    for (uint outputIndex = 0;
                        adapter.EnumOutputs(outputIndex, out var output).Success;
                        outputIndex++)
                    {
                        using (output)
                        {
                            var output1 = output.QueryInterface<IDXGIOutput1>();
                            var outputDesc = output1.Description;
                            var outputBounds = new Vortice.Mathematics.Rect(
                                outputDesc.DesktopCoordinates.Left,
                                outputDesc.DesktopCoordinates.Top,
                                outputDesc.DesktopCoordinates.Right - outputDesc.DesktopCoordinates.Left,
                                outputDesc.DesktopCoordinates.Bottom - outputDesc.DesktopCoordinates.Top);
                            LogManager.Log(LogLevel.Info, $"Found Output {outputIndex}: DeviceName = '{outputDesc.DeviceName.TrimEnd('\0')}', Bounds = {outputBounds}");

                            // Try different matching strategies
                            bool nameMatch = (currentDisplay.DeviceName != null && outputDesc.DeviceName.TrimEnd('\0') == currentDisplay.DeviceName.TrimEnd('\0'));
                            bool boundsMatch = (currentDisplay.Bounds.ToString() == outputBounds.ToString());

                            if (nameMatch || boundsMatch)
                            {
                                targetOutput1 = (IDXGIOutput1)output1.QueryInterface<IDXGIOutput1>();
                                targetAdapter = adapter;
                                foundTarget = true;
                                break;
                            }
                            output1.Dispose();
                        }
                    }

                    if (foundTarget) break;
                }

                // Fallback to specific display index if not found
                if (!foundTarget)
                {
                    int targetIndex = currentDisplay.Index;
                    int currentIndex = 0;

                    for (uint adapterIndex = 0;
                        factory.EnumAdapters1(adapterIndex, out var adapter).Success;
                        adapterIndex++)
                    {
                        for (uint outputIndex = 0;
                            adapter.EnumOutputs(outputIndex, out var output).Success;
                            outputIndex++)
                        {
                            if (currentIndex == targetIndex)
                            {
                                LogManager.Log(LogLevel.Warning, $"Could not match display by name or bounds. Found a fallback index, {targetIndex}.");
                                targetOutput1 = output.QueryInterface<IDXGIOutput1>();
                                targetAdapter = adapter;
                                foundTarget = true;
                                break;
                            }
                            currentIndex++;
                            output.Dispose();
                        }

                        if (foundTarget)
                            break;
                    }
                }

                if (targetAdapter == null || targetOutput1 == null)
                {
                    LogManager.Log(LogLevel.Error, "No suitable display output found for DirectX capture.", true, 6000);
                    throw new Exception("No suitable display output found");
                }

                FeatureLevel[] featureLevels = {
                    FeatureLevel.Level_12_2,
                    FeatureLevel.Level_12_1,
                    FeatureLevel.Level_12_0,
                    FeatureLevel.Level_11_1,
                    FeatureLevel.Level_11_0,
                    FeatureLevel.Level_10_1,
                    FeatureLevel.Level_10_0,
                    FeatureLevel.Level_9_3,
                    FeatureLevel.Level_9_2,
                    FeatureLevel.Level_9_1
                };

                // Create D3D11 device
                var result = D3D11.D3D11CreateDevice(
                    targetAdapter,
                    DriverType.Unknown,
                    DeviceCreationFlags.None,
                    featureLevels,
                    out _dxDevice);

                if (result.Failure || _dxDevice == null)
                {
                    result = D3D11.D3D11CreateDevice(
                      targetAdapter,
                      DriverType.Unknown,
                      DeviceCreationFlags.None,
                      null,
                      out _dxDevice);

                    if (result.Failure || _dxDevice == null)
                    {
                        LogManager.Log(LogLevel.Error, $"Failed to create D3D11 device: {result}", true, 6000);
                        throw new Exception($"Failed to create D3D11 device: {result}");
                    }
                }

                // Create desktop duplication
                _deskDuplication = targetOutput1.DuplicateOutput(_dxDevice);
                _consecutiveFailures = 0;

                LogManager.Log(LogLevel.Info, "DirectX Desktop Duplication initialized successfully.");
            }
            catch (SharpGenException ex) when (ex.ResultCode == Vortice.DXGI.ResultCode.Unsupported || ex.HResult == unchecked((int)0x887A0004))
            {
                LogManager.Log(LogLevel.Error, $"DirectX Desktop Duplication not supported on this system: {ex.Message}", true, 6000);
                _directXFailedPermanently = true;
                DisposeDxgiResources();

                Dictionary.dropdownState["Screen Capture Method"] = "GDI+";
                _currentCaptureMethod = "GDI+";

                LogManager.Log(LogLevel.Error, "DirectX Desktop Duplication not supported on this system. Switched to GDI+ capture.", true, 6000);
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Error, $"Failed to initialize DirectX Desktop Duplication: {ex.Message}", true, 6000);
                DisposeDxgiResources();
                throw;
            }
        }

        public DirectXPointer CaptureDirectXRaw(Rectangle detectionBox)
        {
            int w = detectionBox.Width;
            int h = detectionBox.Height;
            IDXGIResource? desktopResource = null;

            try
            {
                lock (_displayLock)
                {
                    if (_displayChangesPending)
                    {
                        InitializeDxgiDuplication();
                        _displayChangesPending = false;
                    }
                }

                if (_dxDevice == null || _dxDevice.ImmediateContext == null || _deskDuplication == null)
                {
                    InitializeDxgiDuplication();
                    if (_dxDevice == null || _dxDevice.ImmediateContext == null || _deskDuplication == null)
                    {
                        lock (_displayLock) { _displayChangesPending = true; }
                        return default;
                    }
                }

                if (_stagingTex == null || _stagingTex.Description.Width != w || _stagingTex.Description.Height != h)
                {
                    _stagingTex?.Dispose();
                    _stagingTex = _dxDevice.CreateTexture2D(new Texture2DDescription
                    {
                        Width = (uint)w,
                        Height = (uint)h,
                        MipLevels = 1,
                        ArraySize = 1,
                        Format = Format.B8G8R8A8_UNorm,
                        SampleDescription = new(1, 0),
                        Usage = ResourceUsage.Staging,
                        CPUAccessFlags = CpuAccessFlags.Read,
                        BindFlags = BindFlags.None
                    });
                }

                int timeout = _consecutiveFailures > 0 ? 5 : 1;
                var result = _deskDuplication.AcquireNextFrame((uint)timeout, out _, out desktopResource);

                if (result == Vortice.DXGI.ResultCode.WaitTimeout)
                {
                    _consecutiveFailures = 0;
                    return default;
                }
                else if (result == Vortice.DXGI.ResultCode.DeviceRemoved || result == Vortice.DXGI.ResultCode.AccessLost)
                {
                    _consecutiveFailures++;
                    if (_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES)
                        lock (_displayLock) { _displayChangesPending = true; }
                    return default;
                }
                else if (result != Result.Ok)
                {
                    _consecutiveFailures++;
                    return default;
                }

                _consecutiveFailures = 0;

                using var screenTexture = desktopResource.QueryInterface<ID3D11Texture2D>();
                int relativeDetectionLeft = detectionBox.Left - DisplayManager.ScreenLeft;
                int relativeDetectionTop = detectionBox.Top - DisplayManager.ScreenTop;
                
                int srcLeft = Math.Max(relativeDetectionLeft, 0);
                int srcTop = Math.Max(relativeDetectionTop, 0);
                int srcRight = Math.Min(relativeDetectionLeft + w, DisplayManager.ScreenWidth);
                int srcBottom = Math.Min(relativeDetectionTop + h, DisplayManager.ScreenHeight);

                if (srcRight > srcLeft && srcBottom > srcTop)
                {
                    var box = new Box(srcLeft, srcTop, 0, srcRight, srcBottom, 1);
                    _dxDevice.ImmediateContext.CopySubresourceRegion(
                           _stagingTex, 0,
                           (uint)(srcLeft - relativeDetectionLeft),
                           (uint)(srcTop - relativeDetectionTop),
                           0,
                           screenTexture, 0, box);
                }
                else
                {
                    desktopResource.Dispose();
                    _deskDuplication.ReleaseFrame();
                    return default;
                }

                var map = _dxDevice.ImmediateContext.Map(_stagingTex, 0, MapMode.Read, Vortice.Direct3D11.MapFlags.None);
                
                IntPtr ptr = map.DataPointer;
                int stride = (int)map.RowPitch;

                desktopResource.Dispose(); 
                _deskDuplication.ReleaseFrame();

                return new DirectXPointer { Pointer = ptr, Stride = stride };
            }
            catch (Exception e)
            {
                LogManager.Log(LogLevel.Error, $"DirectX Raw capture error: {e.Message}");
                if (++_consecutiveFailures >= MAX_CONSECUTIVE_FAILURES)
                    lock (_displayLock) { _displayChangesPending = true; }
                
                desktopResource?.Dispose();
                return default;
            }
        }

        public void UnmapDirectXRaw()
        {
            if (_dxDevice != null && _stagingTex != null)
            {
                _dxDevice.ImmediateContext.Unmap(_stagingTex, 0);
            }
        }
        
        private Bitmap? DirectX(Rectangle detectionBox)
        {
            var rawPtr = CaptureDirectXRaw(detectionBox);
            if (!rawPtr.IsValid) return GetCachedFrame(detectionBox);

            int w = detectionBox.Width;
            int h = detectionBox.Height;
            
            if (directXBitmap == null || directXBitmap.Width != w || directXBitmap.Height != h)
            {
                directXBitmap?.Dispose();
                directXBitmap = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            }

            var boundsRect = new Rectangle(0, 0, w, h);
            BitmapData mapDest = directXBitmap.LockBits(boundsRect, ImageLockMode.WriteOnly, directXBitmap.PixelFormat);

            try
            {
                unsafe
                {
                    byte* src = (byte*)rawPtr.Pointer;
                    byte* dst = (byte*)mapDest.Scan0;
                    int srcStride = rawPtr.Stride;
                    int dstStride = mapDest.Stride;
                    int copyBytesPerRow = Math.Min(srcStride, dstStride);

                    for (int y = 0; y < h; y++)
                    {
                        Buffer.MemoryCopy(src, dst, dstStride, copyBytesPerRow);
                        src += srcStride;
                        dst += dstStride;
                    }

                    if (Dictionary.toggleState["Third Person Support"])
                    {
                        int width = w / 2;
                        int height = h / 2;
                        int startY = h - height;

                        byte* basePtr = (byte*)mapDest.Scan0;
                        for (int y = startY; y < h; y++)
                        {
                            byte* rowPtr = basePtr + (y * dstStride);
                            for (int x = 0; x < width; x++)
                            {
                                int pixelOffset = x * 4;
                                rowPtr[pixelOffset + 0] = 0;
                                rowPtr[pixelOffset + 1] = 0;
                                rowPtr[pixelOffset + 2] = 0;
                                rowPtr[pixelOffset + 3] = 255;
                            }
                        }
                    }
                }
                
                var result = (Bitmap)directXBitmap.Clone();
                UpdateCache(result, detectionBox);
                return result;
            }
            finally
            {
                directXBitmap.UnlockBits(mapDest);
                _dxDevice?.ImmediateContext.Unmap(_stagingTex, 0);
            }
        }

        #region Frame Caching
        private void UpdateCache(Bitmap frame, Rectangle bounds)
        {
            if (_cachedFrame == null ||
                !_cachedFrameBounds.Equals(bounds) ||
                DateTime.Now - _lastFrameTime > _frameCacheTimeout)
            {
                _cachedFrame?.Dispose();
                _cachedFrame = (Bitmap)frame.Clone();
                _cachedFrameBounds = bounds;
            }
            _lastFrameTime = DateTime.Now;
        }

        private Bitmap? GetCachedFrame(Rectangle detectionBox)
        {
            if (_cachedFrame != null &&
                _cachedFrameBounds.Equals(detectionBox) &&
                DateTime.Now - _lastFrameTime <= _frameCacheTimeout)
            {
                return (Bitmap)_cachedFrame.Clone();
            }
            return null;
        }
        #endregion
        #endregion

        #region GDI
        public Bitmap GDIScreen(Rectangle detectionBox)
        {
            if (_dxDevice != null || _deskDuplication != null)
            {
                DisposeDxgiResources();
            }

            if (screenCaptureBitmap == null || screenCaptureBitmap.Width != detectionBox.Width || screenCaptureBitmap.Height != detectionBox.Height)
            {
                screenCaptureBitmap?.Dispose();
                screenCaptureBitmap = new Bitmap(detectionBox.Width, detectionBox.Height, PixelFormat.Format32bppArgb);
            }

            try
            {
                using (var g = Graphics.FromImage(screenCaptureBitmap))
                {
                    g.CopyFromScreen(
                        detectionBox.Left,
                        detectionBox.Top,
                        0, 0,
                        detectionBox.Size,
                        CopyPixelOperation.SourceCopy
                    );

                    if (Dictionary.toggleState["Third Person Support"])
                    {
                        int width = screenCaptureBitmap.Width / 2;
                        int height = screenCaptureBitmap.Height / 2;
                        int startY = screenCaptureBitmap.Height - height;

                        using (var brush = new SolidBrush(System.Drawing.Color.Black))
                        {
                            g.FillRectangle(brush, 0, startY, width, height);
                        }
                    }
                }

                return screenCaptureBitmap;
            }
            catch (Exception ex)
            {
                LogManager.Log(LogLevel.Error, $"GDI+ screen capture failed: {ex.Message}");
                throw;
            }
        }
        #endregion

        public Bitmap? ScreenGrab(Rectangle detectionBox)
        {
            string selectedMethod = Dictionary.dropdownState["Screen Capture Method"];

            if (_directXFailedPermanently && selectedMethod == "DirectX")
            {
                Dictionary.dropdownState["Screen Capture Method"] = "GDI+";
                selectedMethod = "GDI+";
                _currentCaptureMethod = "GDI+";
            }

            if (selectedMethod != _currentCaptureMethod)
            {
                screenCaptureBitmap?.Dispose();
                screenCaptureBitmap = null;

                directXBitmap?.Dispose();
                directXBitmap = null;

                _currentCaptureMethod = selectedMethod;
                _notificationShown = false;

                if (selectedMethod == "GDI+")
                {
                    DisposeDxgiResources();
                }
                else
                {
                    InitializeDxgiDuplication();
                }
            }

            if (selectedMethod == "DirectX" && !_directXFailedPermanently)
            {
                return DirectX(detectionBox);
            }
            else
            {
                return GDIScreen(detectionBox);
            }
        }

        #region dispose
        public void DisposeDxgiResources()
        {
            lock (_displayLock)
            {
                try
                {
                    if (_deskDuplication != null)
                    {
                        try
                        {
                            _deskDuplication.ReleaseFrame();
                        }
                        catch { }
                    }

                    _deskDuplication?.Dispose();
                    _stagingTex?.Dispose();
                    _dxDevice?.Dispose();
                    _cachedFrame?.Dispose();
                    directXBitmap?.Dispose();

                    _deskDuplication = null;
                    _stagingTex = null;
                    _dxDevice = null;
                    _cachedFrame = null;
                }
                catch (Exception ex)
                {
                    LogManager.Log(LogLevel.Error, $"Error disposing DXGI resources: {ex.Message}");
                }
            }
        }

        public void Dispose()
        {
            DisplayManager.DisplayChanged -= OnDisplayChanged;
            DisposeDxgiResources();
            screenCaptureBitmap?.Dispose();
        }
        #endregion
    }
}