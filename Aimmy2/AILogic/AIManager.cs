using AILogic;
using Aimmy2.Class;
using Class;
using InputLogic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Other;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Visuality;
using static Aimmy2.AILogic.MathUtil;
using static Other.LogManager;

namespace Aimmy2.AILogic
{
    internal class AIManager : IDisposable
    {
        #region Variables
        private readonly object _sizeLock = new object();
        private volatile bool _sizeChangePending = false;

        public void RequestSizeChange(int newSize)
        {
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }
        }
        // Models
        private int IMAGE_SIZE => _currentImageSize;
        private int _currentImageSize;

        private const int SAVE_FRAME_COOLDOWN_MS = 500;
        //predictions
        private DateTime lastSavedTime = DateTime.MinValue;
        private RectangleF LastDetectionBox;
        private KalmanPrediction kalmanPrediction;
        private WiseTheFoxPrediction wtfpredictionManager;

        // Display-aware properties
        private int ScreenWidth => DisplayManager.ScreenWidth;
        private int ScreenHeight => DisplayManager.ScreenHeight;
        private int ScreenLeft => DisplayManager.ScreenLeft;
        private int ScreenTop => DisplayManager.ScreenTop;
        private readonly OverlayManager _overlayManager;

        // Ai loop
        private Task? _aiLoopTask;
        private CancellationTokenSource? _cts;
        private volatile bool _isAiLoopRunning;

        // For Auto-Labelling Data System
        private bool PlayerFound = false;

        // Sticky-Aim 
        private Prediction _currentTarget = null;
        private int _consecutiveFramesWithoutTarget = 0;
        private const int MAX_FRAMES_WITHOUT_TARGET = 2; // Allow 2 frames of target loss

        private double CenterXTranslated = 0;
        private double CenterYTranslated = 0;

        // For Shall0e's Prediction Method
        private int PrevX = 0;
        private int PrevY = 0;

        // Benchmarking
        private int iterationCount = 0;
        private long totalTime = 0;


        //AI Detection Coordinates
        private int detectedX { get; set; }
        private int detectedY { get; set; }

        // current target
        public double AIConf = 0;
        private static int targetX, targetY;

        // Pre-calculated values - now dynamic
        private float _scaleX => ScreenWidth / (float)IMAGE_SIZE;
        private float _scaleY => ScreenHeight / (float)IMAGE_SIZE;

        // Tensor reuse (model inference)
        private float[]? _reusableInputArray;
        private DenseTensor<float>? _reusableTensor;
        private List<NamedOnnxValue>? _reusableInputs;

        private ushort[]? _inputU16Buffer;
        private ushort[]? _outputU16Buffer;
        private float[]? _outputFloatBuffer; // used when output element type == float

        private TensorElementType _modelInputElementType = TensorElementType.Float;
        private TensorElementType _modelOutputElementType = TensorElementType.Float;

        private OrtIoBinding _ioBinding;
        private OrtValue _inputOrtValue;
        private OrtValue _outputOrtValue;
        private bool _ioBindingInitialized = false;
        private readonly object _ioBindingLock = new object();

        // Memory Pinning
        private GCHandle _inputPin;
        private GCHandle _outputFloatPin;
        private GCHandle _outputU16Pin;
        private GCHandle _inputU16Pin;

        // Benchmarking
        private readonly Dictionary<string, BenchmarkData> _benchmarks = new();
        private readonly object _benchmarkLock = new();

        private readonly CaptureManager _captureManager = new();
        private ModelManager _modelManager = new();
        #endregion Variables

        #region Benchmarking

        private class BenchmarkData
        {
            public long TotalTime { get; set; }
            public int CallCount { get; set; }
            public long MinTime { get; set; } = long.MaxValue;
            public long MaxTime { get; set; }
            public double AverageTime => CallCount > 0 ? (double)TotalTime / CallCount : 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private IDisposable Benchmark(string name)
        {
            return new BenchmarkScope(this, name);
        }

        private class BenchmarkScope : IDisposable
        {
            private readonly AIManager _manager;
            private readonly string _name;
            private readonly Stopwatch _sw;

            public BenchmarkScope(AIManager manager, string name)
            {
                _manager = manager;
                _name = name;
                _sw = Stopwatch.StartNew();
            }

            public void Dispose()
            {
                _sw.Stop();
                _manager.RecordBenchmark(_name, _sw.ElapsedMilliseconds);
            }
        }

        private void RecordBenchmark(string name, long elapsedMs)
        {
            lock (_benchmarkLock)
            {
                if (!_benchmarks.TryGetValue(name, out var data))
                {
                    data = new BenchmarkData();
                    _benchmarks[name] = data;
                }

                data.TotalTime += elapsedMs;
                data.CallCount++;
                data.MinTime = Math.Min(data.MinTime, elapsedMs);
                data.MaxTime = Math.Max(data.MaxTime, elapsedMs);
            }
        }

        public void PrintBenchmarks()
        {
            lock (_benchmarkLock)
            {
                var lines = new List<string>
                {
                    "=== AIManager Performance Benchmarks ==="
                };

                foreach (var kvp in _benchmarks.OrderBy(x => x.Key))
                {
                    var data = kvp.Value;
                    lines.Add($"{kvp.Key}: Avg={data.AverageTime:F2}ms, Min={data.MinTime}ms, Max={data.MaxTime}ms, Count={data.CallCount}");
                }
                var overallFps = iterationCount > 0 && totalTime > 0 ? 1000.0 * iterationCount / totalTime : 0;

                lines.Add($"Overall FPS: {overallFps:F2}");

                Log(LogLevel.Info, string.Join(Environment.NewLine, lines));
            }
        }

        #endregion Benchmarking

        public AIManager(string modelPath)
        {
            _overlayManager = new OverlayManager(Dictionary.DetectedPlayerOverlay);
            _modelManager = new();

            _currentImageSize = int.Parse(Dictionary.dropdownState["Image Size"]);

            if (Dictionary.dropdownState["Screen Capture Method"] == "DirectX")
            {
                _captureManager.InitializeDxgiDuplication();
            }

            kalmanPrediction = new KalmanPrediction();
            wtfpredictionManager = new WiseTheFoxPrediction();

            _modelManager.modelOptions = new RunOptions();

            _ = InitializeModel(modelPath);
        }

        #region Models
        public async Task InitializeModel(string modelPath)
        {

            using (Benchmark("ModelInitialization"))
            {
                try
                {
                    await _modelManager.LoadModelAsync(modelPath, IMAGE_SIZE);

                    if (_modelManager.isModelLoaded)
                    {
                        StartAILoop(_modelManager.onnxModel);
                    }
                    else
                    {
                        throw new InvalidOperationException("Model failed to load properly");
                    }

                }
                catch (Exception ex)
                {
                    Log(LogLevel.Error, $"Error starting the model via DirectML: {ex.Message}\n\nFalling back to CPU, performance may be poor.", true);

                    try
                    {
                        await _modelManager.LoadModelAsync(modelPath, IMAGE_SIZE, failure: true);
                        if (_modelManager.isModelLoaded)
                        {
                            StartAILoop(_modelManager.onnxModel);
                        }
                        else
                        {
                            throw new InvalidOperationException("Model failed to load properly");
                        }
                    }
                    catch (Exception e)
                    {
                        Log(LogLevel.Error, $"Error starting the model via CPU: {e.Message}, you won't be able to use aim assist at all.", true);
                    }
                }
                finally
                {
                    FileManager.CurrentlyLoadingModel = false;
                }
            }
        }

        private void InitializeIOBinding(int imageSize)
        {
            if (_ioBindingInitialized &&
                _reusableTensor != null &&
                _reusableTensor.Dimensions[2] == imageSize)
            {
                return;
            }

            lock (_ioBindingLock)
            {
                if (_modelManager.onnxModel == null)
                {
                    return;
                }

                if (Dictionary.dropdownState["Execution Provider"] == "CPU")
                {
                    _ioBindingInitialized = false;
                    return; 
                }
                try
                {
                    _ioBinding?.Dispose();
                    _inputOrtValue?.Dispose();
                    _outputOrtValue?.Dispose();

                    if (_inputPin.IsAllocated) _inputPin.Free();
                    if (_outputFloatPin.IsAllocated) _outputFloatPin.Free();
                    if (_outputU16Pin.IsAllocated) _outputU16Pin.Free();
                    if (_inputU16Pin.IsAllocated) _inputU16Pin.Free();
                }
                catch { }

                _ioBinding = null;
                _inputOrtValue = null;
                _outputOrtValue = null;
                _ioBindingInitialized = false;

                try
                {
                    var inputMeta = _modelManager.onnxModel.InputMetadata;
                    var outputMeta = _modelManager.onnxModel.OutputMetadata;

                    if (inputMeta != null &&
                        inputMeta.TryGetValue(_modelManager.inputName ?? inputMeta.Keys.First(), out var inMeta))
                    {
                        _modelInputElementType = inMeta.ElementDataType;
                    }


                    if (_modelManager.outputNames?.Count > 0 &&
                       outputMeta != null &&
                       outputMeta.TryGetValue(_modelManager.outputNames[0], out var outMeta))
                    {
                        _modelOutputElementType = outMeta.ElementDataType;
                    }

                    Log(LogLevel.Info, $"Model Input element type: {_modelInputElementType}; Output element type: {_modelOutputElementType}");

                    _ioBinding = _modelManager.onnxModel.CreateIoBinding();
                    var memoryInfo = OrtMemoryInfo.DefaultInstance;


                    // INPUT
                    var inputShape = new long[] { 1, 3, imageSize, imageSize };
                    int inputLen = (int)(inputShape.Aggregate(1L, (a, b) => a * b));

                    if (_reusableInputArray == null || _reusableInputArray.Length != inputLen)
                    {
                        if (_inputPin.IsAllocated) _inputPin.Free();
                        _reusableInputArray = new float[inputLen];
                        _inputPin = GCHandle.Alloc(_reusableInputArray, GCHandleType.Pinned);
                        _reusableTensor = null;
                        _reusableInputs = null;
                    }
                    else if (!_inputPin.IsAllocated)
                    {
                        _inputPin = GCHandle.Alloc(_reusableInputArray, GCHandleType.Pinned);
                    }

                    switch (_modelInputElementType)
                    {
                        case TensorElementType.Float:
                            _inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                                memoryInfo, _reusableInputArray, inputShape);
                            _ioBinding.BindInput(_modelManager.inputName ?? "images", _inputOrtValue);
                            Log(LogLevel.Info, "IOBinding: bound pinned float input");
                            break;

                        case TensorElementType.Float16:
                            if (_inputU16Buffer == null || _inputU16Buffer.Length != inputLen)
                            {
                                if (_inputU16Pin.IsAllocated) _inputU16Pin.Free();
                                _inputU16Buffer = new ushort[inputLen];
                                _inputU16Pin = GCHandle.Alloc(_inputU16Buffer, GCHandleType.Pinned);
                            }
                            else if (!_inputU16Pin.IsAllocated)
                            {
                                _inputU16Pin = GCHandle.Alloc(_inputU16Buffer, GCHandleType.Pinned);
                            }

                            _inputOrtValue = OrtValue.CreateTensorValueFromMemory<ushort>(
                                memoryInfo, _inputU16Buffer, inputShape);
                            _ioBinding.BindInput(_modelManager.inputName ?? "images", _inputOrtValue);
                            Log(LogLevel.Info, "IOBinding: bound pinned float16 input");
                            break;

                        default:
                            throw new NotSupportedException(
                                $"Unsupported model input element type: {_modelInputElementType}");
                    }

                    // OUTPUT
                    if (_modelManager.outputNames == null || _modelManager.outputNames.Count == 0)
                        throw new InvalidOperationException("Model output names are not defined.");

                    var outputShape = _modelManager.IsLegacyV8 
                        ? new long[] { 1, _modelManager.NUM_CLASSES + 4, _modelManager.NUM_DETECTIONS }
                        : new long[] { 1, 300, 6 };
                        
                    string outputName = _modelManager.IsLegacyV8 
                        ? _modelManager.outputNames[0] 
                        : (_modelManager.outputNames.Contains("output0") ? "output0" : _modelManager.outputNames[0]);

                    int outputLen = (int)outputShape.Aggregate(1L, (a, b) => a * b);
                    switch (_modelOutputElementType)
                    {
                        case TensorElementType.Float:
                            if (_modelManager.IsLegacyV8)
                            {
                                if (_outputFloatBuffer == null || _outputFloatBuffer.Length != outputLen)
                                {
                                    if (_outputFloatPin.IsAllocated) _outputFloatPin.Free();
                                    _outputFloatBuffer = new float[outputLen];
                                    _outputFloatPin = GCHandle.Alloc(_outputFloatBuffer, GCHandleType.Pinned);
                                }
                                else if (!_outputFloatPin.IsAllocated)
                                {
                                    _outputFloatPin = GCHandle.Alloc(_outputFloatBuffer, GCHandleType.Pinned);
                                }

                                _outputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(
                                    memoryInfo, _outputFloatBuffer, outputShape);
                                _ioBinding.BindOutput(outputName, _outputOrtValue);
                                Log(LogLevel.Info, "IOBinding: bound pinned float output");
                            }
                            else
                            {
                                var providerId = Dictionary.dropdownState["Execution Provider"] == "TensorRT" ? "Tensorrt" : "Cuda";
                                var gpuMemInfo = new OrtMemoryInfo(providerId, OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
                                _ioBinding.BindOutputToDevice(outputName, gpuMemInfo);
                                Log(LogLevel.Info, $"IOBinding: bound float output to Device ({providerId} via ArenaAllocator) for YOLO26");
                            }
                            break;

                        case TensorElementType.Float16:
                            if (_modelManager.IsLegacyV8)
                            {
                                if (_outputU16Buffer == null || _outputU16Buffer.Length != outputLen)
                                {
                                    if (_outputU16Pin.IsAllocated) _outputU16Pin.Free();
                                    _outputU16Buffer = new ushort[outputLen];
                                    _outputU16Pin = GCHandle.Alloc(_outputU16Buffer, GCHandleType.Pinned);
                                }
                                else if (!_outputU16Pin.IsAllocated)
                                {
                                    _outputU16Pin = GCHandle.Alloc(_outputU16Buffer, GCHandleType.Pinned);
                                }

                                _outputOrtValue = OrtValue.CreateTensorValueFromMemory<ushort>(
                                    memoryInfo, _outputU16Buffer, outputShape);
                                _ioBinding.BindOutput(outputName, _outputOrtValue);
                                Log(LogLevel.Info, "IOBinding: bound pinned float16 output");
                            }
                            else
                            {
                                var providerId = Dictionary.dropdownState["Execution Provider"] == "TensorRT" ? "Tensorrt" : "Cuda";
                                var gpuMemInfo = new OrtMemoryInfo(providerId, OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default);
                                _ioBinding.BindOutputToDevice(outputName, gpuMemInfo);
                                Log(LogLevel.Info, $"IOBinding: bound float16 output to Device ({providerId} via ArenaAllocator) for YOLO26");
                            }
                            break;

                        default:
                            throw new NotSupportedException(
                                $"Unsupported model output element type: {_modelOutputElementType}");
                    }

                    _ioBindingInitialized = true;
                }
                catch (Exception ex)
                {
                    try { _ioBinding?.Dispose(); } catch { }
                    try { _inputOrtValue?.Dispose(); } catch { }
                    try { _outputOrtValue?.Dispose(); } catch { }
                    if (_inputPin.IsAllocated) _inputPin.Free();
                    if (_outputFloatPin.IsAllocated) _outputFloatPin.Free();
                    if (_outputU16Pin.IsAllocated) _outputU16Pin.Free();
                    if (_inputU16Pin.IsAllocated) _inputU16Pin.Free();

                    _ioBinding = null;
                    _inputOrtValue = null;
                    _outputOrtValue = null;
                    _ioBindingInitialized = false;

                    Log(LogLevel.Error, $"Failed to initialize IO Binding: {ex.Message}");
                }
            }
        }
        private void StartAILoop(InferenceSession? onnxModel)
        {
            if (onnxModel?.OutputMetadata != null && onnxModel.OutputMetadata.Count > 0)
            {
                _cts?.Cancel();
                try { _aiLoopTask?.Wait(500); } catch { }

                _isAiLoopRunning = true;
                _cts = new CancellationTokenSource();
                _aiLoopTask = Task.Run(() => AiLoop(_cts.Token), _cts.Token);
                Log(LogLevel.Info, "AI loop started");
            }
            else
            {
                Log(LogLevel.Error, "Model not loaded - skipping AI loop start");
            }

        }
        public Dictionary<int, string>? GetModelClasses()
        {
            if (_modelManager.modelClasses != null)
            {
                return _modelManager.modelClasses;
            }

            return null;
        }
        #endregion Models

        #region AI

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldPredict() =>
            Dictionary.toggleState["Show Detected Player"] ||
            Dictionary.toggleState["Constant AI Tracking"] ||
            InputBindingManager.IsHoldingBinding("Aim Keybind") ||
            InputBindingManager.IsHoldingBinding("Second Aim Keybind");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ShouldProcess() =>
            Dictionary.toggleState["Aim Assist"] ||
            Dictionary.toggleState["Show Detected Player"] ||
            Dictionary.toggleState["Auto Trigger"];

        // Performance Metrics
        public struct PerformanceData
        {
            public double Iteration;
            public double ScreenGrab;
            public double Normalization;
            public double Inference;
            public double PrepareKDTree;
            public double HandleAim;
            public double OverallFps;
        }

        public PerformanceData CurrentPerformance { get; private set; }

        public class InferenceMetrics
        {
            private const int BufferSize = 60;
            private readonly double[] _iterationTimes = new double[BufferSize];
            private readonly double[] _screenGrabTimes = new double[BufferSize];
            private readonly double[] _normalizationTimes = new double[BufferSize];
            private readonly double[] _inferenceTimes = new double[BufferSize];
            private readonly double[] _kdTreeTimes = new double[BufferSize];
            private readonly double[] _aimTimes = new double[BufferSize];
            private int _index = 0;
            private int _count = 0;

            public void AddFrame(double iteration, double screenGrab, double norm, double inference, double kdTree, double aim)
            {
                _iterationTimes[_index] = iteration;
                _screenGrabTimes[_index] = screenGrab;
                _normalizationTimes[_index] = norm;
                _inferenceTimes[_index] = inference;
                _kdTreeTimes[_index] = kdTree;
                _aimTimes[_index] = aim;
                _index = (_index + 1) % BufferSize;
                if (_count < BufferSize) _count++;
            }

            public PerformanceData GetAverages()
            {
                if (_count == 0) return new PerformanceData();

                double iteration = 0, screenGrab = 0, norm = 0, inference = 0, kdTree = 0, aim = 0;
                for (int i = 0; i < _count; i++)
                {
                    iteration += _iterationTimes[i];
                    screenGrab += _screenGrabTimes[i];
                    norm += _normalizationTimes[i];
                    inference += _inferenceTimes[i];
                    kdTree += _kdTreeTimes[i];
                    aim += _aimTimes[i];
                }

                iteration /= _count;
                return new PerformanceData
                {
                    Iteration = iteration,
                    ScreenGrab = screenGrab / _count,
                    Normalization = norm / _count,
                    Inference = inference / _count,
                    PrepareKDTree = kdTree / _count,
                    HandleAim = aim / _count,
                    OverallFps = iteration > 0 ? 1000.0 / iteration : 0
                };
            }
        }

        private readonly InferenceMetrics _metrics = new();

        private async Task AiLoop(CancellationToken ct)
        {
            Thread.CurrentThread.Priority = ThreadPriority.Highest;
            DetectedPlayerWindow? DetectedPlayerOverlay = Dictionary.DetectedPlayerOverlay;

            long lastLoopTick = Stopwatch.GetTimestamp();
            double tickToMs = 1000.0 / Stopwatch.Frequency;

            while (!ct.IsCancellationRequested && _isAiLoopRunning)
            {
                lock (_sizeLock)
                {
                    if (_sizeChangePending)
                    {
                        continue;
                    }
                }

                long loopStartTick = Stopwatch.GetTimestamp();
                double iterationTime = (loopStartTick - lastLoopTick) * tickToMs;
                lastLoopTick = loopStartTick;

                _captureManager.HandlePendingDisplayChanges();

                using (Benchmark("AILoopIteration"))
                {
                    _overlayManager.UpdateFOV();

                    if (ShouldProcess())
                    {
                        if (ShouldPredict())
                        {
                            Prediction? closestPrediction;
                            
                            // We capture timings directly within GetClosestPrediction or wrap it
                            // Since GetClosestPrediction is where ScreenGrab/Norm/Inference happen:
                            long screenGrabTicks = 0;
                            long normTicks = 0;
                            long inferenceTicks = 0;
                            long kdTreeTicks = 0;
                            long aimTicks = 0;

                            closestPrediction = GetClosestPrediction(out screenGrabTicks, out normTicks, out inferenceTicks, out kdTreeTicks);

                            if (closestPrediction == null)
                            {
                                _overlayManager.DisableOverlay(DetectedPlayerOverlay!);
                                _metrics.AddFrame(iterationTime, screenGrabTicks * tickToMs, normTicks * tickToMs, inferenceTicks * tickToMs, kdTreeTicks * tickToMs, 0);
                                CurrentPerformance = _metrics.GetAverages();
                                continue;
                            }

                            using (Benchmark("AutoTrigger"))
                            {
                                await AutoTrigger();
                            }

                            using (Benchmark("CalculateCoordinates"))
                            {
                                CalculateCoordinates(DetectedPlayerOverlay, closestPrediction, _scaleX, _scaleY);
                            }

                            long aimStart = Stopwatch.GetTimestamp();
                            using (Benchmark("HandleAim"))
                            {
                                HandleAim(closestPrediction);
                            }
                            aimTicks = Stopwatch.GetTimestamp() - aimStart;

                            _metrics.AddFrame(iterationTime, screenGrabTicks * tickToMs, normTicks * tickToMs, inferenceTicks * tickToMs, kdTreeTicks * tickToMs, aimTicks * tickToMs);
                            CurrentPerformance = _metrics.GetAverages();

                            totalTime += (long)((Stopwatch.GetTimestamp() - loopStartTick) * tickToMs);
                            iterationCount++;
                        }
                        else
                        {
                            await Task.Delay(1, ct);
                        }
                    }
                    else
                    {
                        await Task.Delay(1, ct);
                    }
                }
            }
        }

        #region AI Loop Functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task AutoTrigger()
        {
            if (!Dictionary.toggleState["Auto Trigger"] ||
                !(InputBindingManager.IsHoldingBinding("Aim Keybind") && !InputBindingManager.IsHoldingBinding("Second Aim Keybind")) ||
                Dictionary.toggleState["Constant AI Tracking"])
            {
                CheckSprayRelease();
                return;
            }

            if (Dictionary.toggleState["Spray Mode"])
            {
                await MouseManager.DoTriggerClick(LastDetectionBox);
                return;
            }


            if (Dictionary.toggleState["Cursor Check"])
            {
                var mousePos = WinAPICaller.GetCursorPosition();

                if (!DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePos.X, mousePos.Y)))
                {
                    return;
                }

                if (LastDetectionBox.Contains(mousePos.X, mousePos.Y))
                {
                    await MouseManager.DoTriggerClick(LastDetectionBox);
                }
            }
            else
            {
                await MouseManager.DoTriggerClick();
            }

            if (!Dictionary.toggleState["Aim Assist"] || !Dictionary.toggleState["Show Detected Player"]) return;

        }
        private void CheckSprayRelease()
        {
            if (!Dictionary.toggleState["Spray Mode"]) return;

            bool shouldSpray = Dictionary.toggleState["Auto Trigger"] &&
                (InputBindingManager.IsHoldingBinding("Aim Keybind") && InputBindingManager.IsHoldingBinding("Second Aim Keybind")); 

            if (!shouldSpray)
            {
                MouseManager.ResetSprayState();
            }
        }

        private void CalculateCoordinates(DetectedPlayerWindow DetectedPlayerOverlay, Prediction closestPrediction, float scaleX, float scaleY)
        {
            AIConf = closestPrediction.Confidence;

            if (Dictionary.toggleState["Show Detected Player"] && Dictionary.DetectedPlayerOverlay != null)
            {
                using (Benchmark("UpdateOverlay"))
                {
                    _overlayManager.UpdateOverlay(closestPrediction, LastDetectionBox, AIConf);
                }
                if (!Dictionary.toggleState["Aim Assist"]) return;
            }

            double YOffset = Dictionary.sliderSettings["Y Offset (Up/Down)"];
            double XOffset = Dictionary.sliderSettings["X Offset (Left/Right)"];

            double YOffsetPercentage = Dictionary.sliderSettings["Y Offset (%)"];
            double XOffsetPercentage = Dictionary.sliderSettings["X Offset (%)"];

            var rect = closestPrediction.Rectangle;

            if (Dictionary.toggleState["X Axis Percentage Adjustment"])
            {
                detectedX = (int)((rect.X + (rect.Width * (XOffsetPercentage / 100))) * scaleX);
            }
            else
            {
                detectedX = (int)((rect.X + rect.Width / 2) * scaleX + XOffset);
            }

            if (Dictionary.toggleState["Y Axis Percentage Adjustment"])
            {
                detectedY = (int)((rect.Y + rect.Height - (rect.Height * (YOffsetPercentage / 100))) * scaleY + YOffset);
            }
            else
            {
                detectedY = CalculateDetectedY(scaleY, YOffset, closestPrediction);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CalculateDetectedY(float scaleY, double YOffset, Prediction closestPrediction)
        {
            var rect = closestPrediction.Rectangle;
            float yBase = rect.Y;
            float yAdjustment = 0;

            switch (Dictionary.dropdownState["Aiming Boundaries Alignment"])
            {
                case "Center":
                    yAdjustment = rect.Height / 2;
                    break;

                case "Top":
                    break;

                case "Bottom":
                    yAdjustment = rect.Height;
                    break;
            }

            return (int)((yBase + yAdjustment) * scaleY + YOffset);
        }

        private void HandleAim(Prediction closestPrediction)
        {
            if (Dictionary.toggleState["Aim Assist"] &&
                (Dictionary.toggleState["Constant AI Tracking"] ||
                 Dictionary.toggleState["Aim Assist"] && InputBindingManager.IsHoldingBinding("Aim Keybind") ||
                 Dictionary.toggleState["Aim Assist"] && InputBindingManager.IsHoldingBinding("Second Aim Keybind")))
            {
                if (Dictionary.toggleState["Predictions"])
                {
                    HandlePredictions(kalmanPrediction, closestPrediction, detectedX, detectedY);
                }
                else
                {
                    MouseManager.MoveCrosshair(detectedX, detectedY);
                }
            }
        }

        private void HandlePredictions(KalmanPrediction kalmanPrediction, Prediction closestPrediction, int detectedX, int detectedY)
        {
            var predictionMethod = Dictionary.dropdownState["Prediction Method"];
            switch (predictionMethod)
            {
                case "Kalman Filter":
                    KalmanPrediction.Detection detection = new()
                    {
                        X = detectedX,
                        Y = detectedY,
                        Timestamp = DateTime.UtcNow
                    };

                    kalmanPrediction.UpdateKalmanFilter(detection);
                    var predictedPosition = kalmanPrediction.GetKalmanPosition();

                    MouseManager.MoveCrosshair(predictedPosition.X, predictedPosition.Y);
                    break;

                case "Shall0e's Prediction":
                    ShalloePredictionV2.xValues.Add(detectedX - PrevX);
                    ShalloePredictionV2.yValues.Add(detectedY - PrevY);

                    if (ShalloePredictionV2.xValues.Count > 5)
                    {
                        ShalloePredictionV2.xValues.RemoveAt(0);
                        ShalloePredictionV2.yValues.RemoveAt(0);
                    }

                    MouseManager.MoveCrosshair(ShalloePredictionV2.GetSPX(), detectedY);

                    PrevX = detectedX;
                    PrevY = detectedY;
                    break;

                case "wisethef0x's EMA Prediction":
                    WiseTheFoxPrediction.WTFDetection wtfdetection = new()
                    {
                        X = detectedX,
                        Y = detectedY,
                        Timestamp = DateTime.UtcNow
                    };

                    wtfpredictionManager.UpdateDetection(wtfdetection);
                    var wtfpredictedPosition = wtfpredictionManager.GetEstimatedPosition();

                    MouseManager.MoveCrosshair(wtfpredictedPosition.X, detectedY);
                    break;
            }
        }

        private Prediction? GetClosestPrediction(out long screenGrabTicks, out long normTicks, out long inferenceTicks, out long kdTreeTicks, bool useMousePosition = true)
        {
            screenGrabTicks = 0;
            normTicks = 0;
            inferenceTicks = 0;
            kdTreeTicks = 0;

            if (Dictionary.dropdownState["Detection Area Type"] == "Closest to Mouse")
            {
                var mousePos = WinAPICaller.GetCursorPosition();

                if (DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePos.X, mousePos.Y)))
                {
                    targetX = (int)mousePos.X;
                    targetY = (int)mousePos.Y;
                }
                else
                {
                    targetX = (int)(DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2));
                    targetY = (int)(DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2));
                }
            }
            else
            {
                targetX = (int)(DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2));
                targetY = (int)(DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2));
            }

            Rectangle detectionBox = new(targetX - IMAGE_SIZE / 2, targetY - IMAGE_SIZE / 2, IMAGE_SIZE, IMAGE_SIZE);
            Bitmap? frame = null;

            using (Benchmark("BitmapToFloatArray"))
            {
                int requiredLength = 3 * IMAGE_SIZE * IMAGE_SIZE;
                if (_reusableInputArray == null || _reusableInputArray.Length != requiredLength)
                {
                    InitializeIOBinding(IMAGE_SIZE);
                }

                if (Dictionary.dropdownState["Screen Capture Method"] == "DirectX")
                {
                    using (Benchmark("ScreenGrab"))
                    {
                        long sgStart = Stopwatch.GetTimestamp();
                        var rawPtr = _captureManager.CaptureDirectXRaw(detectionBox);
                        screenGrabTicks = Stopwatch.GetTimestamp() - sgStart;

                        if (rawPtr.IsValid)
                        {
                            long normStart = Stopwatch.GetTimestamp();
                            unsafe
                            {
                                NormalizeDirect((byte*)rawPtr.Pointer, rawPtr.Stride, _reusableInputArray!, IMAGE_SIZE);
                            }
                            normTicks = Stopwatch.GetTimestamp() - normStart;

                            _captureManager.UnmapDirectXRaw();
                            goto InferenceStart;
                        }
                    }
                }

                using (Benchmark("ScreenGrab"))
                {
                    long sgStart = Stopwatch.GetTimestamp();
                    frame = _captureManager.ScreenGrab(detectionBox);
                    screenGrabTicks = Stopwatch.GetTimestamp() - sgStart;
                }

                if (frame == null) return null;

                long normStartManual = Stopwatch.GetTimestamp();
                BitmapToFloatArrayInPlace(frame, _reusableInputArray!, IMAGE_SIZE);
                normTicks = Stopwatch.GetTimestamp() - normStartManual;
            }

        InferenceStart:

            if (_modelManager.onnxModel == null)
            {
                frame?.Dispose();
                return null;
            }

            if (!_ioBindingInitialized ||
                _reusableTensor == null ||
                _reusableTensor.Dimensions[2] != IMAGE_SIZE)
            {
                using (Benchmark("IOBindingInitialization"))
                {
                    InitializeIOBinding(IMAGE_SIZE);
                }
            }

            Tensor<float>? outputTensor = null;
            using (Benchmark("ModelInference"))
            {
                long infStart = Stopwatch.GetTimestamp();
                try
                {
                    if (_ioBindingInitialized && _inputOrtValue != null && _outputOrtValue != null)
                    {
                        if (_modelInputElementType == TensorElementType.Float16)
                        {
                            int len = _reusableInputArray!.Length;
                            var inputU16 = _inputU16Buffer!;
                            for (int i = 0; i < len; i++)
                            {
                                float v = _reusableInputArray[i];
                                inputU16[i] = FloatToHalfBits(v);
                            }
                        }

                        _modelManager.onnxModel.RunWithBinding(_modelManager.modelOptions, _ioBinding);

                        var outputShape = _modelManager.IsLegacyV8 
                            ? new int[] { 1, _modelManager.NUM_CLASSES + 4, _modelManager.NUM_DETECTIONS }
                            : new int[] { 1, 300, 6 };

                        if (_modelManager.IsLegacyV8)
                        {
                            if (_modelOutputElementType == TensorElementType.Float)
                            {
                                outputTensor = new DenseTensor<float>(_outputFloatBuffer!, outputShape);
                            }
                            else if (_modelOutputElementType == TensorElementType.Float16)
                            {
                                var outU16 = _outputU16Buffer!;
                                var outFloat = new float[outU16.Length];
                                for (int i = 0; i < outU16.Length; i++)
                                    outFloat[i] = HalfBitsToFloat(outU16[i]);

                                outputTensor = new DenseTensor<float>(outFloat, outputShape);
                            }
                        }
                        else
                        {
                            using var ortValues = _ioBinding.GetOutputValues();
                            var gpuOutput = ortValues[0];
                            
                            if (_modelOutputElementType == TensorElementType.Float16)
                            {
                                var outU16 = gpuOutput.GetTensorDataAsSpan<ushort>();
                                var outFloat = new float[outU16.Length];
                                for (int i = 0; i < outU16.Length; i++)
                                    outFloat[i] = HalfBitsToFloat(outU16[i]);

                                outputTensor = new DenseTensor<float>(outFloat, new int[] { 1, 300, 6 });
                            }
                            else
                            {
                                var outSpan = gpuOutput.GetTensorDataAsSpan<float>();
                                outputTensor = new DenseTensor<float>(outSpan.ToArray(), new int[] { 1, 300, 6 });
                            }
                        }
                    }
                    else
                    {
                        if (_reusableTensor == null || _reusableTensor.Dimensions[2] != IMAGE_SIZE)
                        {
                            _reusableTensor = new DenseTensor<float>(_reusableInputArray, new int[] { 1, 3, IMAGE_SIZE, IMAGE_SIZE });

                            if (_reusableInputs == null)
                                _reusableInputs = new List<NamedOnnxValue>(1);

                            _reusableInputs.Clear();
                            _reusableInputs.Add(NamedOnnxValue.CreateFromTensor(_modelManager.inputName ?? "images", _reusableTensor));
                        }
                        else
                        {
                            _reusableInputArray!.AsSpan().CopyTo(_reusableTensor.Buffer.Span);
                        }

                        using var results = _modelManager.onnxModel.Run(_reusableInputs, _modelManager.outputNames, _modelManager.modelOptions);
                        outputTensor = results[0].AsTensor<float>();
                    }
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Error, $"Inference error: {ex.Message}");
                    _ioBindingInitialized = false; 
                }
                inferenceTicks = Stopwatch.GetTimestamp() - infStart;
            }

            if (outputTensor == null)
            {
                Log(LogLevel.Error, "Model inference returned null output tensor.", true, 2000);
                SaveFrame(frame);
                return null;
            }

            float FovSize = (float)Dictionary.sliderSettings["FOV Size"];
            float fovMinX = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxX = (IMAGE_SIZE + FovSize) / 2.0f;
            float fovMinY = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxY = (IMAGE_SIZE + FovSize) / 2.0f;

            List<Prediction> KDPredictions;
            long kdStart = Stopwatch.GetTimestamp();
            using (Benchmark("PrepareKDTreeData")) 
            {
                KDPredictions = PrepareKDTreeData(outputTensor, detectionBox, fovMinX, fovMaxX, fovMinY, fovMaxY);
            }
            kdTreeTicks = Stopwatch.GetTimestamp() - kdStart;

            if (KDPredictions.Count == 0)
            {
                SaveFrame(frame);
                frame?.Dispose();
                return null;
            }

            Prediction? bestCandidate = null;
            double bestDistSq = double.MaxValue;
            double center = IMAGE_SIZE / 2.0;

            using (Benchmark("LinearSearch"))
            {
                foreach (var p in KDPredictions)
                {
                    var dx = p.CenterXTranslated * IMAGE_SIZE - center; 
                    var dy = p.CenterYTranslated * IMAGE_SIZE - center;
                    double d2 = dx * dx + dy * dy; 

                    if (d2 < bestDistSq) { bestDistSq = d2; bestCandidate = p; }
                }
            }

            Prediction? finalTarget = HandleStickyAim(bestCandidate, KDPredictions);
            if (finalTarget != null)
            {
                UpdateDetectionBox(finalTarget, detectionBox);
                SaveFrame(frame, finalTarget);
                frame?.Dispose();
                return finalTarget;
            }

            frame?.Dispose();
            return null;
        }

        private Prediction? HandleStickyAim(Prediction? bestCandidate, List<Prediction> KDPredictions)
        {
            if (!Dictionary.toggleState["Sticky Aim"])
            {
                _currentTarget = bestCandidate; 
                return bestCandidate;
            }

            float threshold = (float)Dictionary.sliderSettings["Sticky Aim Threshold"];
            float thresholdSqr = threshold * threshold;

            if (bestCandidate == null || KDPredictions == null || KDPredictions.Count == 0)
            {
                if (_currentTarget != null)
                {
                    if (++_consecutiveFramesWithoutTarget > MAX_FRAMES_WITHOUT_TARGET)
                    {
                        return null;
                    }
                    return _currentTarget;
                }
                return null;
            }
            _consecutiveFramesWithoutTarget = 0;

            if (_currentTarget != null)
            {
                Prediction? matchedTarget = null;
                float minSqrDistance = float.MaxValue;

                foreach (var candidate in KDPredictions)
                {
                    float sqrDistance = Distance(_currentTarget, candidate);
                    if (sqrDistance < minSqrDistance && sqrDistance < thresholdSqr)
                    {
                        minSqrDistance = sqrDistance;
                        matchedTarget = candidate;
                    }
                }

                if (matchedTarget != null)
                {
                    _consecutiveFramesWithoutTarget = 0;
                    _currentTarget = matchedTarget;
                    return matchedTarget;
                }
            }

            _currentTarget = bestCandidate;
            return bestCandidate;
        }

        private readonly List<Prediction> _kdPredictions = new(8400);
        private List<Prediction> PrepareKDTreeData(
            Tensor<float> outputTensor,
            Rectangle detectionBox,
            float fovMinX, float fovMaxX, float fovMinY, float fovMaxY)
        {
            _kdPredictions.Clear();
            float minConfidence = (float)Dictionary.sliderSettings["AI Minimum Confidence"] / 100.0f;
            string selectedClass = Dictionary.dropdownState["Target Class"];
            int selectedClassId = -1;

            int numDetections = _modelManager.NUM_DETECTIONS;
            int numClasses = _modelManager.NUM_CLASSES;
            var modelClasses = _modelManager.modelClasses;

            if (selectedClass != "Best Confidence")
            {
                foreach (var kv in modelClasses)
                {
                    if (kv.Value == selectedClass) { selectedClassId = kv.Key; break; }
                }
            }


            if (_modelManager.IsLegacyV8)
            {
                for (int i = 0; i < numDetections; i++)
                {
                    float x_center = outputTensor[0, 0, i];
                    float y_center = outputTensor[0, 1, i];
                    float width = outputTensor[0, 2, i];
                    float height = outputTensor[0, 3, i];

                    int bestClassId = 0;
                    float bestConfidence = 0f;

                    if (numClasses == 1)
                    {
                        bestConfidence = outputTensor[0, 4, i];
                    }
                    else
                    {
                        if (selectedClassId == -1)
                        {
                            for (int classId = 0; classId < numClasses; classId++)
                            {
                                float classConfidence = outputTensor[0, 4 + classId, i];
                                if (classConfidence > bestConfidence)
                                {
                                    bestConfidence = classConfidence;
                                    bestClassId = classId;
                                }
                            }
                        }
                        else
                        {
                            bestConfidence = outputTensor[0, 4 + selectedClassId, i];
                            bestClassId = selectedClassId;
                        }
                    }

                    if (bestConfidence < minConfidence) continue;

                    float x_min = x_center - width / 2;
                    float y_min = y_center - height / 2;
                    float x_max = x_center + width / 2;
                    float y_max = y_center + height / 2;

                    if (x_min < fovMinX || x_max > fovMaxX || y_min < fovMinY || y_max > fovMaxY) continue;

                    RectangleF rect = new(x_min, y_min, width, height);
                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = bestConfidence,
                        ClassId = bestClassId,
                        ClassName = modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                        CenterXTranslated = x_center / IMAGE_SIZE,
                        CenterYTranslated = y_center / IMAGE_SIZE,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    _kdPredictions.Add(prediction);
                }
            }
            else
            {
                var denseOut = outputTensor as DenseTensor<float>;
                ReadOnlySpan<float> span = denseOut != null ? denseOut.Buffer.Span : outputTensor.ToArray().AsSpan();
                
                float inputRes = _modelManager.ModelFixedSize > 0 ? _modelManager.ModelFixedSize : IMAGE_SIZE;
                float scale = IMAGE_SIZE;

                for (int i = 0; i < 300; i++)
                {
                    int offset = i * 6;
                    float score = span[offset + 4];
                    if (score < minConfidence) continue;

                    float x1 = (span[offset + 0] / inputRes) * scale;
                    float y1 = (span[offset + 1] / inputRes) * scale;
                    float x2 = (span[offset + 2] / inputRes) * scale;
                    float y2 = (span[offset + 3] / inputRes) * scale;
                    int classId = (int)span[offset + 5];

                    float width = x2 - x1;
                    float height = y2 - y1;
                    float x_center = x1 + width / 2f;
                    float y_center = y1 + height / 2f;

                    if (x_center - width / 2 < fovMinX || x_center + width / 2 > fovMaxX || 
                        y_center - height / 2 < fovMinY || y_center + height / 2 > fovMaxY) continue;

                    RectangleF rect = new(x1, y1, width, height);

                    Prediction prediction = new()
                    {
                        Rectangle = rect,
                        Confidence = score,
                        ClassId = classId,
                        ClassName = modelClasses.GetValueOrDefault(classId, $"Class_{classId}"),
                        CenterXTranslated = x_center / scale,
                        CenterYTranslated = y_center / scale,
                        ScreenCenterX = detectionBox.Left + x_center,
                        ScreenCenterY = detectionBox.Top + y_center
                    };

                    _kdPredictions.Add(prediction);
                }
            }

            return _kdPredictions;
        }
        private void UpdateDetectionBox(Prediction target, Rectangle detectionBox)
        {
            float translatedXMin = target.Rectangle.X + detectionBox.Left;
            float translatedYMin = target.Rectangle.Y + detectionBox.Top;
            LastDetectionBox = new(translatedXMin, translatedYMin,
                target.Rectangle.Width, target.Rectangle.Height);

            CenterXTranslated = target.CenterXTranslated;
            CenterYTranslated = target.CenterYTranslated;
        }
        #endregion AI Loop Functions

        #endregion AI

        #region Screen Capture
        private void SaveFrame(Bitmap frame, Prediction? DoLabel = null)
        {
            if (!Dictionary.toggleState["Collect Data While Playing"]) return;

            if (Dictionary.toggleState["Constant AI Tracking"] && !Dictionary.toggleState["Auto Label Data"]) return;

            if ((DateTime.Now - lastSavedTime).TotalMilliseconds < SAVE_FRAME_COOLDOWN_MS) return;

            lastSavedTime = DateTime.Now;
            string uuid = Guid.NewGuid().ToString();

            string imagePath = Path.Combine("bin", "images", $"{uuid}.jpg");

            frame?.Save(imagePath, ImageFormat.Jpeg);

            if (Dictionary.toggleState["Auto Label Data"] && DoLabel != null)
            {
                var labelPath = Path.Combine("bin", "labels", $"{uuid}.txt");

                float x = (DoLabel!.Rectangle.X + DoLabel.Rectangle.Width / 2) / (frame?.Width ?? 1);
                float y = (DoLabel!.Rectangle.Y + DoLabel.Rectangle.Height / 2) / (frame?.Height ?? 1);
                float width = DoLabel.Rectangle.Width / (frame?.Width ?? 1);
                float height = DoLabel.Rectangle.Height / (frame?.Height ?? 1);

                File.WriteAllText(labelPath, $"{DoLabel.ClassId} {x} {y} {width} {height}");
            }
        }
        #endregion Screen Capture


        public void Dispose()
        {
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }

            _isAiLoopRunning = false;
            try
            {
                _cts?.Cancel();
                if (_aiLoopTask != null)
                {
                    _aiLoopTask.Wait(TimeSpan.FromSeconds(1));
                }
            }
            catch { }

            PrintBenchmarks();

            _captureManager.Dispose();

            _reusableInputArray = null;
            _reusableInputs = null;
            _reusableTensor = null;
            _cts?.Dispose();

            if (_inputPin.IsAllocated) _inputPin.Free();
            if (_outputFloatPin.IsAllocated) _outputFloatPin.Free();
            if (_outputU16Pin.IsAllocated) _outputU16Pin.Free();
            if (_inputU16Pin.IsAllocated) _inputU16Pin.Free();

            _modelManager.Dispose();
        }
    }
    public class Prediction
    {
        public RectangleF Rectangle { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; } = 0;
        public string ClassName { get; set; } = "Enemy";
        public float CenterXTranslated { get; set; }
        public float CenterYTranslated { get; set; }
        public float ScreenCenterX { get; set; }  
        public float ScreenCenterY { get; set; }
    }
}