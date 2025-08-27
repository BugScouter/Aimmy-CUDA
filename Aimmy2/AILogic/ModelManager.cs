using Aimmy2.Class;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json.Linq;
using Other;
using System.IO;
using System.Windows;
using static Aimmy2.AILogic.MathUtil;
using static Other.LogManager;

namespace Aimmy2.AILogic
{
    public class ModelManager
    {
        public RunOptions? modelOptions { get; set; }
        public InferenceSession? onnxModel { get; private set; }
        public List<string>? outputNames { get; private set; }
        public string? inputName { get; private set; }
        public List<string>? inputNames { get; private set; }


        public int NUM_DETECTIONS { get; set; } = 8400; // Will be set dynamically for dynamic models
        public bool IsDynamicModel { get; set; } = false;
        public int ModelFixedSize { get; set; } = 640; // Store the fixed size for non-dynamic models
        public int NUM_CLASSES { get; set; } = 1;

        public Dictionary<int, string> modelClasses { get; private set; } = new()
        {
            { 0, "enemy" }
        };

        public static event Action<Dictionary<int, string>>? ClassesUpdated;
        public static event Action<int>? ImageSizeUpdated;

        public bool isModelLoaded => onnxModel != null && outputNames != null && outputNames.Count > 0;



        public async Task LoadModelAsync(string modelPath, int IMAGE_SIZE, bool failure = false) // default value for failure is false, obviously
        {
            try
            {
                using SessionOptions sessionOptions = new()
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = false,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_PARALLEL,
                    IntraOpNumThreads = 1,
                    InterOpNumThreads = 1
                };

                if (!failure)
                {
                    switch (Dictionary.dropdownState["Execution Provider"])
                    {
                        //TODO: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
                        case "TensorRT":
                            var tensorrtOptions = new OrtTensorRTProviderOptions();

                            tensorrtOptions.UpdateOptions(new Dictionary<string, string>
                        {
                            { "device_id", "0" }, // 1 for true 0 for false
                            { "trt_fp16_enable", "1" },
                            //{ "trt_int8_enable", "1" },
                            { "trt_engine_cache_enable", "1" },
                            { "trt_engine_cache_path", "bin/tensorrt_cache" }
                        });

                            Log(LogLevel.Info, $"{modelPath} {Path.ChangeExtension(modelPath, ".engine")}");
                            Log(LogLevel.Info, "Loading model with TensorRT, expect long model load time.", true, 15000);

                            sessionOptions.AppendExecutionProvider_Tensorrt(tensorrtOptions);
                            break;
                        case "CUDA":
                            var cudaProviderOptions = new OrtCUDAProviderOptions();

                            cudaProviderOptions.UpdateOptions(new Dictionary<string, string>
                        {
                            { "device_id", "0" },
                            { "arena_extend_strategy", "kNextPowerOfTwo" },
                            { "do_copy_in_default_stream", "1" },

                        });
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
                onnxModel = await Task.Run(() => new InferenceSession(modelPath, sessionOptions), cts.Token);
                //_onnxModel = new InferenceSession(modelPath, sessionOptions);
                outputNames = new(onnxModel.OutputMetadata.Keys);
                inputNames = onnxModel.InputMetadata.Keys.ToList();
                inputName = inputNames.FirstOrDefault(); // pick the first one, safe default

                Log(LogLevel.Info, $"Model loaded successfully: {modelPath}");
                // Validate the onnx model output shape (ensure model is OnnxV8)
                if (!ValidateOnnxShape(IMAGE_SIZE))
                {
                    onnxModel?.Dispose();
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
                onnxModel?.Dispose();
                throw;
            }
            #endregion
            //return Task.CompletedTask;
        }

        #region model shape validation and class loading
        public bool ValidateOnnxShape(int IMAGE_SIZE)
        {
            if (onnxModel != null)
            {
                var inputMetadata = onnxModel.InputMetadata;
                var outputMetadata = onnxModel.OutputMetadata;

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
                    var supportedSizes = new[] { "640", "512", "416", "320", "256", "160" }; //isnt it 128?
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
        public void LoadClasses()
        {
            if (onnxModel == null) return;
            modelClasses.Clear();

            try
            {
                var metadata = onnxModel.ModelMetadata;

                if (metadata != null && metadata.CustomMetadataMap.TryGetValue("names", out string? value) && !string.IsNullOrEmpty(value))
                {
                    JObject data = JObject.Parse(value);
                    if (data != null && data.Type == JTokenType.Object)
                    {
                        foreach (var item in data)
                        {
                            if (int.TryParse(item.Key, out int classId) && item.Value.Type == JTokenType.String)
                            {
                                modelClasses[classId] = item.Value.ToString();
                            }
                        }
                        NUM_CLASSES = modelClasses.Count > 0 ? modelClasses.Keys.Max() + 1 : 1;
                        Log(LogLevel.Info, $"Loaded {modelClasses.Count} class(es) from model metadata: {data.ToString(Newtonsoft.Json.Formatting.None)}", false);
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
                ClassesUpdated?.Invoke(new Dictionary<int, string>(modelClasses));
            }
            catch (Exception ex)
            {
                Log(LogLevel.Error, $"Error loading classes: {ex.Message}", true);
            }
        }
#endregion
        public void Dispose()
        {
            try
            {
                onnxModel?.Dispose();
                onnxModel = null;
            }
            catch (Exception ex)
            {
                Log(LogLevel.Warning, $"Error disposing ONNX model: {ex.Message}", false);
            }

            try
            {
                modelOptions?.Dispose();
                modelOptions = null;
            }
            catch (Exception ex)
            {
                Log(LogLevel.Warning, $"Error disposing model options: {ex.Message}", false);
            }

            // Clear collections
            outputNames?.Clear();
            outputNames = null;

            modelClasses?.Clear();

            Log(LogLevel.Info, "ModelManager disposed successfully.", false);
        }
    }
}
