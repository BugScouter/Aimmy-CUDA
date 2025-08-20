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

                //File.WriteAllLines("AIManager_Benchmarks.txt", lines);

                Log(LogLevel.Info, string.Join(Environment.NewLine, lines));
            }
        }

        #endregion Benchmarking

        public AIManager(string modelPath)
        {
            _overlayManager = new OverlayManager(Dictionary.DetectedPlayerOverlay);
            _modelManager = new();

            // Initialize the cached image size
            _currentImageSize = int.Parse(Dictionary.dropdownState["Image Size"]);

            // Initialize DXGI capture for current display
            if (Dictionary.dropdownState["Screen Capture Method"] == "DirectX")
            {
                _captureManager.InitializeDxgiDuplication();
            }

            kalmanPrediction = new KalmanPrediction();
            wtfpredictionManager = new WiseTheFoxPrediction();

            _modelManager.modelOptions = new RunOptions();


            // Attempt to load via CUDA (else fallback to CPU)
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
        private void StartAILoop(InferenceSession? onnxModel)
        {
            if (onnxModel?.OutputMetadata != null && onnxModel.OutputMetadata.Count > 0)
            {
                Log(LogLevel.Info, "Starting AI Loop", false);

                // Ensure we stop any existing loop before starting a new one
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

        private async Task AiLoop(CancellationToken ct)
        {
            Stopwatch stopwatch = new();
            DetectedPlayerWindow? DetectedPlayerOverlay = Dictionary.DetectedPlayerOverlay;

            while (!ct.IsCancellationRequested && _isAiLoopRunning)
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
                                closestPrediction = GetClosestPrediction();
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
                            await Task.Delay(1, ct);
                        }
                    }
                    else
                    {
                        // No work to do—sleep briefly to free up CPU
                        await Task.Delay(1, ct);
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

        private Prediction? GetClosestPrediction(bool useMousePosition = true)
        {
            //int adjustedTargetX, adjustedTargetY;

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
                int requiredLength = 3 * IMAGE_SIZE * IMAGE_SIZE;

                if (_reusableInputArray == null || _reusableInputArray.Length != requiredLength)
                {
                    _reusableInputArray = new float[requiredLength];
                    _reusableTensor = null;
                    _reusableInputs = null;
                }
                inputArray = _reusableInputArray;

                BitmapToFloatArrayInPlace(frame, inputArray, IMAGE_SIZE);
            }

            // Reuse tensor and inputs - recreate if size changed
            if (_reusableTensor == null || _reusableTensor.Dimensions[2] != IMAGE_SIZE)
            {
                _reusableTensor = new DenseTensor<float>(inputArray, new int[] { 1, 3, IMAGE_SIZE, IMAGE_SIZE });
                _reusableInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", _reusableTensor) };
            }
            //else
            //{
            //    // Directly copy into existing DenseTensor buffer
            //    inputArray.AsSpan().CopyTo(_reusableTensor.Buffer.Span);
            //}

            if (_modelManager.onnxModel == null)
            {
                frame.Dispose();
                return null; // Model not loaded, exit early
            }

            //IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
            Tensor<float>? outputTensor = null;
            using (Benchmark("ModelInference"))
            {
                using var results = _modelManager.onnxModel.Run(_reusableInputs, _modelManager.outputNames, _modelManager.modelOptions);
                outputTensor = results[0].AsTensor<float>();
            }

            if (outputTensor == null)
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

            //List<double[]> KDpoints;
            List<Prediction> KDPredictions;
            using (Benchmark("PrepareKDTreeData"))
            {
                 KDPredictions = PrepareKDTreeData(outputTensor, detectionBox, fovMinX, fovMaxX, fovMinY, fovMaxY);
            }

            if (KDPredictions.Count == 0)
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
                frame.Dispose();
                return finalTarget;
            }

            frame.Dispose();
            return null;
        }

        // sticky aim needs to be refined
        // this is a very basic implementation of sticky aim, it will be improved in the future.
        /// TODO: REFINE linear search to find closest target based on mouse position / current target (?)
        /// e.g whatever is closer to the current target
        private Prediction? HandleStickyAim(Prediction? bestCandidate, List<Prediction> KDPredictions)
        {
            if (!Dictionary.toggleState["Sticky Aim"])
            {
                _currentTarget = bestCandidate; // update anyway
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

                    // keep previous target while within grace period
                    return _currentTarget;
                }
                return null;
            }
            // reset consecutive frames since we have a target
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

            // acquire a new target
            _currentTarget = bestCandidate;
            return bestCandidate;
        }

        
        private List<Prediction> PrepareKDTreeData(
            Tensor<float> outputTensor,
            Rectangle detectionBox,
            float fovMinX, float fovMaxX, float fovMinY, float fovMaxY)
        {
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


            //var KDpoints = new List<double[]>(_modelManager.NUM_DETECTIONS); // Pre-allocate with estimated capacity
            var KDpredictions = new List<Prediction>(numDetections);

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

                //KDpoints.Add(new double[] { x_center, y_center });
                KDpredictions.Add(prediction);
            }

            return KDpredictions;
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
            try
            {
                _cts?.Cancel();
                if (_aiLoopTask != null)
                {
                    _aiLoopTask.Wait(TimeSpan.FromSeconds(1));
                }
            }
            catch { }

            // Print final benchmarks
            PrintBenchmarks();

            // Dispose DXGI objects
            _captureManager.Dispose();

            // Clean up other resources
            _reusableInputArray = null;
            _reusableInputs = null;
            _reusableTensor = null;
            _cts?.Dispose();

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
        public float ScreenCenterX { get; set; }  // Absolute screen position
        public float ScreenCenterY { get; set; }
    }
}