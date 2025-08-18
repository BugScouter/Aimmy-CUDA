using AILogic;
using Aimmy2.Class;
using Class;
using InputLogic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;
using Other;
using Supercluster.KDTree;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.CompilerServices;
using System.Windows;
using Visuality;
using static Other.LogManager;
using static Aimmy2.AILogic.MathUtil;

namespace Aimmy2.AILogic
{
    internal class AIManager : IDisposable
    {
        #region Variables

        private int _currentImageSize;
        private readonly object _sizeLock = new object();
        private volatile bool _sizeChangePending = false;

        public void RequestSizeChange(int newSize)
        {
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }
        }

        // Dynamic properties instead of constants
        public int IMAGE_SIZE => _currentImageSize;
        private int NUM_DETECTIONS { get; set; } = 8400; // Will be set dynamically for dynamic models
        private bool IsDynamicModel { get; set; } = false;
        private int ModelFixedSize { get; set; } = 640; // Store the fixed size for non-dynamic models
        private int NUM_CLASSES { get; set; } = 1;

        private Dictionary<int, string> _modelClasses = new Dictionary<int, string>
        {
            { 0, "enemy" }
        };

        public Dictionary<int, string> ModelClasses => _modelClasses; // apparently this is better than making _modelClasses public

        public static event Action<Dictionary<int, string>>? ClassesUpdated;
        public static event Action<int>? ImageSizeUpdated;

        private const int SAVE_FRAME_COOLDOWN_MS = 500;

        private DateTime lastSavedTime = DateTime.MinValue;
        private List<string>? _outputNames;
        private RectangleF LastDetectionBox;
        private KalmanPrediction kalmanPrediction;
        private WiseTheFoxPrediction wtfpredictionManager;

        

        // Display-aware properties
        private int ScreenWidth => DisplayManager.ScreenWidth;
        private int ScreenHeight => DisplayManager.ScreenHeight;
        private int ScreenLeft => DisplayManager.ScreenLeft;
        private int ScreenTop => DisplayManager.ScreenTop;
        private readonly OverlayManager _overlayManager;

        private readonly RunOptions? _modeloptions;
        private InferenceSession? _onnxModel;

        private Thread? _aiLoopThread;
        private volatile bool _isAiLoopRunning;

        // For Auto-Labelling Data System
        private bool PlayerFound = false;

        // Sticky-Aim 
        private Prediction _currentTarget = null;
        private int _consecutiveFramesWithoutTarget = 0;
        private const int MAX_FRAMES_WITHOUT_TARGET = 3; // Allow 3 frames of target loss
        //private const float TARGET_MATCH_THRESHOLD = 50f; (now is a slider, Dictionary.sliderSettings["Sticky Aim Threshold"])

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
        private DenseTensor<float>? _reusableTensor;
        private float[]? _reusableInputArray;
        private List<NamedOnnxValue>? _reusableInputs;

        // Benchmarking
        private readonly Dictionary<string, BenchmarkData> _benchmarks = new();
        private readonly object _benchmarkLock = new();


        private readonly CaptureManager _captureManager = new();

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

                //File.WriteAllLines("AIManager_Benchmarks.txt", lines);

                Log(LogLevel.Info, string.Join(Environment.NewLine, lines));
            }
        }

        #endregion Benchmarking

        public AIManager(string modelPath)
        {
            _overlayManager = new OverlayManager(Dictionary.DetectedPlayerOverlay);

            // Initialize the cached image size
            _currentImageSize = int.Parse(Dictionary.dropdownState["Image Size"]);

            // Initialize DXGI capture for current display
            if (Dictionary.dropdownState["Screen Capture Method"] == "DirectX")
            {
                _captureManager.InitializeDxgiDuplication();
            }

            kalmanPrediction = new KalmanPrediction();
            wtfpredictionManager = new WiseTheFoxPrediction();

            _modeloptions = new RunOptions();


            // Attempt to load via CUDA (else fallback to CPU)
            _ = InitializeModel(modelPath);
        }

        #region Models
        private async Task InitializeModel(string modelPath)
        {
            using (Benchmark("ModelInitialization"))
            {
                try
                {
                    await LoadModelAsync(modelPath);
                }
                catch (Exception ex)
                {
                    Log(LogLevel.Error, $"Error starting the model via DirectML: {ex.Message}\n\nFalling back to CPU, performance may be poor.", true);

                    try
                    {
                        await LoadModelAsync(modelPath, failure: true);
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

        private async Task<Task> LoadModelAsync(string modelPath, bool failure = false) // default value for failure is false, obviously
        {
            try
            {
                using SessionOptions sessionOptions = new()
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = false,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_PARALLEL
                };

                if (!failure)
                {
                    switch (Dictionary.dropdownState["Execution Provider"])
                    {
                        case "TensorRT":
                            var tensorrtOptions = new OrtTensorRTProviderOptions();

                            tensorrtOptions.UpdateOptions(new Dictionary<string, string>
                        {
                            { "device_id", "0" }, // 1 for true 0 for false
                            { "trt_fp16_enable", "1" },
                            { "trt_engine_cache_enable", "1" },
                            { "trt_engine_cache_path", "bin/tensorrt_cache" }
                        });

                            Log(LogLevel.Info, $"{modelPath} {Path.ChangeExtension(modelPath, ".engine")}");
                            Log(LogLevel.Info, "Loading model with TensorRT, expect long model load time.", true, 15000);

                            sessionOptions.AppendExecutionProvider_Tensorrt(tensorrtOptions);
                            break;
                        case "CUDA":
                            Log(LogLevel.Info, "Loading model with CUDA execution provider.", false);
                            sessionOptions.AppendExecutionProvider_CUDA();
                            break;
                        default:
                            Log(LogLevel.Info, "Loading model with CPU execution provider.", false);
                            sessionOptions.AppendExecutionProvider_CPU(); // Fallback to CPU if no other provider is selected
                            break;
                    }
                }
                else
                {
                    sessionOptions.AppendExecutionProvider_CPU();
                }

                var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
                _onnxModel = await Task.Run(() => new InferenceSession(modelPath, sessionOptions), cts.Token);
                //_onnxModel = new InferenceSession(modelPath, sessionOptions);
                _outputNames = new List<string>(_onnxModel.OutputMetadata.Keys);

                Log(LogLevel.Info, $"Model loaded successfully: {modelPath}");
                // Validate the onnx model output shape (ensure model is OnnxV8)
                if (!ValidateOnnxShape())
                {
                    _onnxModel?.Dispose();
                    //return; // Exit early if validation fails
                }
            }
            #region Handling Exceptions 
            //(There are many precautions here because users are not very careful with their installations)
            catch (OnnxRuntimeException ex)
            {
                string? message = null, title = null;

                bool hasTensorRTError = ex.Message.Contains("TensorRT execution provider is not enabled in this build") ||
                                        (ex.Message.Contains("LoadLibrary failed with error 126") && ex.Message.Contains("onnxruntime_providers_tensorrt.dll"));

                bool hasCUDAError = ex.Message.Contains("CUDA execution provider is not enabled in this build") ||
                                    (ex.Message.Contains("LoadLibrary failed with error 126") && ex.Message.Contains("onnxruntime_providers_cuda.dll"));

                if (hasTensorRTError)
                {
                    if (RequirementsManager.IsTensorRTInstalled())
                    {
                        message = "TensorRT has been found by Aimmy, but not by ONNX. Please check your configuration.\nHint: Check CUDNN and your CUDA, and install dependencies to PATH correctly.";
                        title = "Configuration Error";
                    }
                    else
                    {
                        message = "TensorRT execution provider has not been found on your build. Please check your configuration.\nHint: Download TensorRT 10.3.x and install the LIB folder to PATH.";
                        title = "TensorRT Error";
                    }
                }
                else if (hasCUDAError)
                {
                    if (RequirementsManager.IsCUDAInstalled() && RequirementsManager.IsCUDNNInstalled())
                    {
                        message = "CUDA & CUDNN have been found by Aimmy, but not by ONNX. Please check your configuration.\nHint: Check CUDNN and your CUDA installations, path, etc. PATH directories should point directly towards the DLLs.";
                        title = "Configuration Error";
                    }
                    else
                    {
                        message = "CUDA execution provider has not been found on your build. Please check your configuration.\nHint: Download CUDA 12.x. Then install CUDNN 9.x to your PATH (or install the DLL included with Aimmy).";
                        title = "CUDA Error";
                    }
                }

                if (message != null)
                {
                    MessageBox.Show(message, title!, MessageBoxButton.OK, MessageBoxImage.Error);
                    Log(LogLevel.Error, message);
                }

            }
            catch (Exception ex)
            {
                Log(LogLevel.Error, $"Error loading the model: {ex.Message}", true);
                _onnxModel?.Dispose();
            }
            #endregion
            finally
            {
                if (_onnxModel?.OutputMetadata != null && _onnxModel.OutputMetadata.Count > 0)
                {
                    Log(LogLevel.Info, "Starting AI Loop", false);
                    _isAiLoopRunning = true;
                    _aiLoopThread = new Thread(AiLoop) 
                    {
                        IsBackground = true,
                        Priority = ThreadPriority.AboveNormal
                    };
                    _aiLoopThread.Start();
                    // Begin the loop
                }
                else
                {
                    Log(LogLevel.Error, "Model not loaded - skipping AI loop start");
                }
            }

            return Task.CompletedTask;
        }

        private int CalculateNumDetections(int imageSize)
        {
            // YOLOv8 detection calculation: (size/8)² + (size/16)² + (size/32)²
            int stride8 = imageSize / 8;
            int stride16 = imageSize / 16;
            int stride32 = imageSize / 32;

            return (stride8 * stride8) + (stride16 * stride16) + (stride32 * stride32);
        }

        private bool ValidateOnnxShape()
        {
            if (_onnxModel != null)
            {
                var inputMetadata = _onnxModel.InputMetadata;
                var outputMetadata = _onnxModel.OutputMetadata;

                Log(LogLevel.Info, "=== Model Metadata ===");
                Log(LogLevel.Info, "Input Metadata:");

                bool isDynamic = false;
                int fixedInputSize = 0;

                foreach (var kvp in inputMetadata)
                {
                    string dimensionsStr = string.Join("x", kvp.Value.Dimensions);
                    Log(LogLevel.Info, $"  Name: {kvp.Key}, Dimensions: {dimensionsStr}");

                    // Check if model is dynamic (dimensions are -1)
                    if (kvp.Value.Dimensions.Any(d => d == -1))
                    {
                        isDynamic = true;
                    }
                    else if (kvp.Value.Dimensions.Length == 4)
                    {
                        // For fixed models, check if it's the expected format (1x3xHxW)
                        fixedInputSize = kvp.Value.Dimensions[2]; // Height should equal Width for square models
                    }
                }

                Log(LogLevel.Info, "Output Metadata:");
                foreach (var kvp in outputMetadata)
                {
                    string dimensionsStr = string.Join("x", kvp.Value.Dimensions);
                    Log(LogLevel.Info, $"  Name: {kvp.Key}, Dimensions: {dimensionsStr}");
                }

                IsDynamicModel = isDynamic;

                if (IsDynamicModel)
                {
                    // For dynamic models, calculate NUM_DETECTIONS based on selected image size
                    NUM_DETECTIONS = CalculateNumDetections(IMAGE_SIZE);
                    LoadClasses();
                    ImageSizeUpdated?.Invoke(IMAGE_SIZE);
                    Log(LogLevel.Info, $"Loaded dynamic model - using selected image size {IMAGE_SIZE}x{IMAGE_SIZE} with {NUM_DETECTIONS} detections", true, 3000);
                }
                else
                {
                    // For fixed models, auto-adjust image size if needed
                    ModelFixedSize = fixedInputSize;

                    // List of supported sizes
                    var supportedSizes = new[] { "640", "512", "416", "320", "256", "160" };
                    var fixedSizeStr = fixedInputSize.ToString();

                    if (fixedInputSize != IMAGE_SIZE && supportedSizes.Contains(fixedSizeStr))
                    {
                        // Auto-adjust the image size to match the model
                        Log(LogLevel.Warning,
                            $"Fixed-size model expects {fixedInputSize}x{fixedInputSize}. Automatically adjusting Image Size setting.",
                            true, 3000);

                        Dictionary.dropdownState["Image Size"] = fixedSizeStr;

                        // Update the UI dropdown if it exists
                        Application.Current?.Dispatcher.BeginInvoke(() =>
                        {
                            try
                            {
                                // Find the MainWindow and update the dropdown
                                var mainWindow = Application.Current.Windows.OfType<MainWindow>().FirstOrDefault();
                                if (mainWindow?.SettingsMenuControlInstance != null)
                                {
                                    mainWindow.SettingsMenuControlInstance.UpdateImageSizeDropdown(fixedSizeStr);
                                }
                            }
                            catch { }
                        });

                        // The IMAGE_SIZE property will now return the correct value
                        NUM_DETECTIONS = CalculateNumDetections(fixedInputSize);
                        ImageSizeUpdated?.Invoke(fixedInputSize);
                    }
                    else if (!supportedSizes.Contains(fixedSizeStr))
                    {
                        Log(LogLevel.Error,
                            $"Model requires unsupported size {fixedInputSize}x{fixedInputSize}. Supported sizes are: {string.Join(", ", supportedSizes)}",
                            true, 10000);
                        return false;
                    }

                    LoadClasses();

                    // For static models, validate the expected shape
                    var expectedShape = new int[] { 1, 4 + NUM_CLASSES, NUM_DETECTIONS };
                    if (!outputMetadata.Values.All(metadata => metadata.Dimensions.SequenceEqual(expectedShape)))
                    {
                        Log(LogLevel.Error,
                            $"Output shape does not match the expected shape of {string.Join("x", expectedShape)}.\nThis model will not work with Aimmy, please use an YOLOv8 model converted to ONNXv8.",
                            true, 10000);
                        return false;
                    }

                    Log(LogLevel.Info, $"Loaded fixed-size model: {fixedInputSize}x{fixedInputSize}", true, 2000);
                }

                return true;
            }

            return false;
        }

        private void LoadClasses()
        {
            if (_onnxModel == null) return;
            _modelClasses.Clear();

            try
            {
                var metadata = _onnxModel.ModelMetadata;

                if (metadata != null && metadata.CustomMetadataMap.TryGetValue("names", out string? value) && !string.IsNullOrEmpty(value))
                {
                    JObject data = JObject.Parse(value);
                    if (data != null && data.Type == JTokenType.Object)
                    {
                        foreach (var item in data)
                        {
                            if (int.TryParse(item.Key, out int classId) && item.Value.Type == JTokenType.String)
                            {
                                _modelClasses[classId] = item.Value.ToString();
                            }
                        }
                        NUM_CLASSES = _modelClasses.Count > 0 ? _modelClasses.Keys.Max() + 1 : 1;
                        Log(LogLevel.Info, $"Loaded {_modelClasses.Count} class(es) from model metadata: {data.ToString(Newtonsoft.Json.Formatting.None)}", false);
                    }
                    else
                    {
                        Log(LogLevel.Error, "Model metadata 'names' field is not a valid JSON object.", true);
                    }
                }
                else
                {
                    Log(LogLevel.Error, "Model metadata does not contain 'names' field for classes.", true);
                }
                ClassesUpdated?.Invoke(new Dictionary<int, string>(_modelClasses));
            }
            catch (Exception ex)
            {
                Log(LogLevel.Error, $"Error loading classes: {ex.Message}", true);
            }
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

        private async void AiLoop() 
        {
            Stopwatch stopwatch = new();
            DetectedPlayerWindow? DetectedPlayerOverlay = Dictionary.DetectedPlayerOverlay;

            while (_isAiLoopRunning)
            {
                // Check for pending size changes at the start of each iteration
                lock (_sizeLock)
                {
                    if (_sizeChangePending)
                    {
                        // Skip this iteration to allow clean shutdown
                        continue;
                    }
                }

                stopwatch.Restart();
                _captureManager.HandlePendingDisplayChanges();

                using (Benchmark("AILoopIteration"))
                {
                    _overlayManager.UpdateFOV();

                    if (ShouldProcess())
                    {
                        if (ShouldPredict())
                        {
                            Prediction? closestPrediction;
                            using (Benchmark("GetClosestPrediction"))
                            {
                                closestPrediction = await GetClosestPrediction();
                            }

                            if (closestPrediction == null)
                            {
                                _overlayManager.DisableOverlay(DetectedPlayerOverlay!);
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

                            using (Benchmark("HandleAim"))
                            {
                                HandleAim(closestPrediction);
                            }

                            totalTime += stopwatch.ElapsedMilliseconds;
                            iterationCount++;
                        }
                        else
                        {
                            // Processing so we are at the ready but not holding right/click.
                            await Task.Delay(1);
                        }
                    }
                    else
                    {
                        // No work to do—sleep briefly to free up CPU
                        await Task.Delay(1);
                    }
                }

                stopwatch.Stop();
            }
        }

        #region AI Loop Functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task AutoTrigger()
        {
            // if auto trigger is disabled,
            // or if the aim keybinds are not held,
            // or if constant AI tracking is enabled,
            // we check for spray release and return
            if (!Dictionary.toggleState["Auto Trigger"] ||
                !(InputBindingManager.IsHoldingBinding("Aim Keybind") && !InputBindingManager.IsHoldingBinding("Second Aim Keybind")) ||
                Dictionary.toggleState["Constant AI Tracking"]) // this logic is a bit weird, but it works.
                                                                // but it might need to be revised
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

            // if auto trigger is disabled, we reset the spray state
            // if the aim keybinds are not held, we reset the spray state
            bool shouldSpray = Dictionary.toggleState["Auto Trigger"] &&
                (InputBindingManager.IsHoldingBinding("Aim Keybind") && InputBindingManager.IsHoldingBinding("Second Aim Keybind")); //||
                                                                                                                                     //Dictionary.toggleState["Constant AI Tracking"];

            // spray mode might need to be revised.
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
                    // yBase is already at the top
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

        private async Task<Prediction?> GetClosestPrediction(bool useMousePosition = true)
        {
            int adjustedTargetX, adjustedTargetY;

            if (Dictionary.dropdownState["Detection Area Type"] == "Closest to Mouse")
            {
                var mousePos = WinAPICaller.GetCursorPosition();

                // Check if mouse is on the current display
                if (DisplayManager.IsPointInCurrentDisplay(new System.Windows.Point(mousePos.X, mousePos.Y)))
                {
                    // Mouse is on current display, use its position
                    targetX = mousePos.X;
                    targetY = mousePos.Y;
                }
                else
                {
                    // Mouse is on different display, use center of current display
                    targetX = DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2);
                    targetY = DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2);
                }
            }
            else
            {
                // Center of current display
                targetX = DisplayManager.ScreenLeft + (DisplayManager.ScreenWidth / 2);
                targetY = DisplayManager.ScreenTop + (DisplayManager.ScreenHeight / 2);
            }

            Rectangle detectionBox = new(targetX - IMAGE_SIZE / 2, targetY - IMAGE_SIZE / 2, IMAGE_SIZE, IMAGE_SIZE); // Detection box dynamic size

            Bitmap? frame;

            using (Benchmark("ScreenGrab"))
            {
                frame = _captureManager.ScreenGrab(detectionBox);
            }

            if (frame == null) return null;

            float[] inputArray;
            using (Benchmark("BitmapToFloatArray"))
            {
                if (_reusableInputArray == null || _reusableInputArray.Length != 3 * IMAGE_SIZE * IMAGE_SIZE)
                {
                    _reusableInputArray = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
                }
                inputArray = _reusableInputArray;

                // Fill the reusable array
                BitmapToFloatArrayInPlace(frame, inputArray, IMAGE_SIZE);
            }

            // Reuse tensor and inputs - recreate if size changed
            if (_reusableTensor == null || _reusableTensor.Dimensions[2] != IMAGE_SIZE)
            {
                _reusableTensor = new DenseTensor<float>(inputArray, new int[] { 1, 3, IMAGE_SIZE, IMAGE_SIZE });
                _reusableInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", _reusableTensor) };
            }
            else
            {
                // Directly copy into existing DenseTensor buffer
                inputArray.AsSpan().CopyTo(_reusableTensor.Buffer.Span);
            }

            if (_onnxModel == null) return null;

            //IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            Tensor<float>? outputTensor = null;
            using (Benchmark("ModelInference"))
            {
                using var results = _onnxModel.Run(_reusableInputs, _outputNames, _modeloptions);
                outputTensor = results[0].AsTensor<float>();
            }

            if(outputTensor == null)
            {
                Log(LogLevel.Error, "Model inference returned null output tensor.", true, 2000);
                SaveFrame(frame);
                return null;
            }

            // Calculate the FOV boundaries
            float FovSize = (float)Dictionary.sliderSettings["FOV Size"];
            float fovMinX = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxX = (IMAGE_SIZE + FovSize) / 2.0f;
            float fovMinY = (IMAGE_SIZE - FovSize) / 2.0f;
            float fovMaxY = (IMAGE_SIZE + FovSize) / 2.0f;

            List<double[]> KDpoints;
            List<Prediction> KDPredictions;
            using (Benchmark("PrepareKDTreeData"))
            {
                (KDpoints, KDPredictions) = PrepareKDTreeData(outputTensor, detectionBox, fovMinX, fovMaxX, fovMinY, fovMaxY);
            }
            
            if (KDpoints.Count == 0 || KDPredictions.Count == 0)
            {
                SaveFrame(frame);
                return null;
            }
            // i removed kd tree.
            Prediction? bestCandidate = null;
            double bestDistSq = double.MaxValue;
            double center = IMAGE_SIZE / 2.0;

            using (Benchmark("LinearSearch"))
            {
                foreach (var p in KDPredictions)
                {
                    var dx = p.CenterXTranslated * IMAGE_SIZE - center; // or use x_center from KDpoints
                    var dy = p.CenterYTranslated * IMAGE_SIZE - center;
                    double d2 = dx * dx + dy * dy; // dx^2 + dy^2

                    if (d2 < bestDistSq) { bestDistSq = d2; bestCandidate = p; }
                }
            }

            Prediction? finalTarget = HandleStickyAim(bestCandidate, KDPredictions);
            if (finalTarget != null)
            {
                UpdateDetectionBox(finalTarget, detectionBox);
                SaveFrame(frame, finalTarget);
                return finalTarget;
            }

            frame.Dispose();
            return null;
        }
        private Prediction? HandleStickyAim(Prediction? bestCandidate, List<Prediction> KDPredictions)
        {
            bool stickyAimEnabled = Dictionary.toggleState["Sticky Aim"];
            if (!stickyAimEnabled)
            {
                _currentTarget = bestCandidate; // update anyway
                return bestCandidate;
            }

            float thresholdSqr = (float)Math.Pow(Dictionary.sliderSettings["Sticky Aim Threshold"], 2);

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

                if (++_consecutiveFramesWithoutTarget > MAX_FRAMES_WITHOUT_TARGET)
                {
                    _currentTarget = null;
                } else
                {
                    return null; // No match found, keep the current target
                }
            }

            // acquire a new target
            _currentTarget = bestCandidate;
            return bestCandidate;
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
        private (List<double[]>, List<Prediction>) PrepareKDTreeData(
            Tensor<float> outputTensor, 
            Rectangle detectionBox,
            float fovMinX, float fovMaxX, float fovMinY, float fovMaxY)
        {
            float minConfidence = (float)Dictionary.sliderSettings["AI Minimum Confidence"] / 100.0f;
            string selectedClass = Dictionary.dropdownState["Target Class"];
            int selectedClassId = selectedClass == "Best Confidence" ? -1 : _modelClasses.FirstOrDefault(c => c.Value == selectedClass).Key;

            var KDpoints = new List<double[]>(NUM_DETECTIONS); // Pre-allocate with estimated capacity
            var KDpredictions = new List<Prediction>(NUM_DETECTIONS);

            for (int i = 0; i < NUM_DETECTIONS; i++)
            {
                float x_center = outputTensor[0, 0, i];
                float y_center = outputTensor[0, 1, i];
                float width = outputTensor[0, 2, i];
                float height = outputTensor[0, 3, i];

                int bestClassId = 0;
                float bestConfidence = 0f;

                if (NUM_CLASSES == 1)
                {
                    bestConfidence = outputTensor[0, 4, i];
                }
                else
                {
                    if (selectedClassId == -1)
                    {
                        for (int classId = 0; classId < NUM_CLASSES; classId++)
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
                    ClassName = _modelClasses.GetValueOrDefault(bestClassId, $"Class_{bestClassId}"),
                    CenterXTranslated = x_center / IMAGE_SIZE, // !! CenterXTranslated is normalized to [0, 1]
                    CenterYTranslated = y_center / IMAGE_SIZE,
                    //CenterXTranslated = (x_center - detectionBox.Left) / IMAGE_SIZE,
                    //CenterYTranslated = (y_center - detectionBox.Top) / IMAGE_SIZE,
                    ScreenCenterX = detectionBox.Left + x_center,
                    ScreenCenterY = detectionBox.Top + y_center
                };

                KDpoints.Add(new double[] { x_center, y_center });
                KDpredictions.Add(prediction);
            }

            return (KDpoints, KDpredictions);
        }

        #endregion AI Loop Functions

        #endregion AI

        #region Screen Capture
        private void SaveFrame(Bitmap frame, Prediction? DoLabel = null)
        {
            // Only save frames if "Collect Data While Playing" is enabled
            if (!Dictionary.toggleState["Collect Data While Playing"]) return;

            // Skip if we're in constant tracking mode (unless auto-labeling is enabled)
            if (Dictionary.toggleState["Constant AI Tracking"] && !Dictionary.toggleState["Auto Label Data"]) return;

            // Cooldown check
            if ((DateTime.Now - lastSavedTime).TotalMilliseconds < SAVE_FRAME_COOLDOWN_MS) return;

            lastSavedTime = DateTime.Now;
            string uuid = Guid.NewGuid().ToString();

            // Clone the bitmap to avoid threading issues
            string imagePath = Path.Combine("bin", "images", $"{uuid}.jpg");

            // Save synchronously to avoid "Object is currently in use elsewhere" error
            frame.Save(imagePath, ImageFormat.Jpeg);

            if (Dictionary.toggleState["Auto Label Data"] && DoLabel != null)
            {
                var labelPath = Path.Combine("bin", "labels", $"{uuid}.txt");

                float x = (DoLabel!.Rectangle.X + DoLabel.Rectangle.Width / 2) / frame.Width;
                float y = (DoLabel!.Rectangle.Y + DoLabel.Rectangle.Height / 2) / frame.Height;
                float width = DoLabel.Rectangle.Width / frame.Width;
                float height = DoLabel.Rectangle.Height / frame.Height;

                File.WriteAllText(labelPath, $"{DoLabel.ClassId} {x} {y} {width} {height}");
            }
        }
        #endregion Screen Capture


        public void Dispose()
        {
            // Signal that we're shutting down
            lock (_sizeLock)
            {
                _sizeChangePending = true;
            }

            // Stop the loop
            _isAiLoopRunning = false;
            if (_aiLoopThread != null && _aiLoopThread.IsAlive)
            {
                if (!_aiLoopThread.Join(TimeSpan.FromSeconds(1)))
                {
                    try { _aiLoopThread.Interrupt(); }
                    catch { }
                }
            }

            // Print final benchmarks
            PrintBenchmarks();

            // Dispose DXGI objects
            _captureManager.Dispose();

            // Clean up other resources
            _reusableInputArray = null;
            _reusableInputs = null;
            _onnxModel?.Dispose();
            _modeloptions?.Dispose();
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
        public float ScreenCenterX { get; set; }  // Absolute screen position
        public float ScreenCenterY { get; set; }
    }
}